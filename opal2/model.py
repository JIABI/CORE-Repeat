"""Conditional set-to-joint-distribution model in fixed measured coordinates.

R2 uses a conditional latent neural process with a chemical Gaussian prior,
an amortized context-conditional prior, and a training-only variational target
posterior. The predictive decoder is linear in its latent variables and is
marginalized exactly. Shared factors describe covariance, not identified causal
biological/site effects. The original R1 project remains a separate artifact.
An optional chemistry-response kernel augments the prior. The copula_t4
observation family maps this same latent Gaussian dependence into Student-t
margins with a full Jacobian; it is trained by predictive likelihood, not the
legacy Gaussian ELBO.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
from numbers import Integral
import math
import torch
from torch import nn
from torch.nn import functional as F

from .kernels import MeasurementKernelOperator


class EnvironmentNoiseCache:
    """Reusable environmental Monte Carlo draws across compound chunks.

    Keys encode hierarchy level and all known ancestor IDs. Reuse one cache for
    one Monte Carlo evaluation, then discard it for independent replicates.
    Local compound/residual/diagonal draws are deliberately not cached. Unknown
    group IDs are not globally identifiable and therefore are never cached.
    """
    def __init__(self):
        self._entries: dict[tuple, torch.Tensor] = {}

    def draw(self, key: tuple, n_samples: int, rank: int, *, device: torch.device,
             dtype: torch.dtype, generator: torch.Generator | None = None) -> torch.Tensor:
        if key in self._entries:
            value = self._entries[key]
            if (value.shape != (n_samples, rank) or value.dtype != dtype
                    or value.device != torch.device(device)):
                raise ValueError("Environmental noise cache shape/dtype/device mismatch")
            return value
        value = torch.randn(n_samples, rank, device=device, dtype=dtype, generator=generator)
        self._entries[key] = value
        return value

    def clear(self) -> None:
        self._entries.clear()


class GroupedProfileEncoder(nn.Module):
    """Coordinate-complete group tokens with multi-layer inter-group attention.

    Groups are actual named CellProfiler families; every coordinate is retained.
    Attention mixes group tokens before a learned query pools them. This is not
    scalar attention pooling over independently encoded groups.
    """
    def __init__(self, feature_groups: Mapping[str, Sequence[int]], hidden_dim: int = 256,
                 attention_layers: int = 2, attention_heads: int = 4):
        super().__init__()
        self.feature_groups = {str(k): list(map(int, v)) for k, v in feature_groups.items()}
        flat = [i for values in self.feature_groups.values() for i in values]
        if not flat or sorted(flat) != list(range(len(flat))):
            raise ValueError("Feature groups must partition coordinates 0,...,D-1 exactly")
        if any(not group for group in self.feature_groups.values()):
            raise ValueError("Feature groups cannot be empty")
        self.feature_dim, self.hidden_dim = len(flat), hidden_dim
        self.group_names = list(self.feature_groups)
        if attention_layers < 0 or hidden_dim % attention_heads:
            raise ValueError("Nonnegative attention layers and a head-divisible hidden dimension are required")
        self.attention_layers, self.attention_heads = attention_layers, attention_heads
        token_dim = hidden_dim
        self.group_nets = nn.ModuleList()
        for k, ids in enumerate(self.feature_groups.values()):
            self.register_buffer(f"indices_{k}", torch.tensor(ids, dtype=torch.long))
            self.group_nets.append(nn.Sequential(nn.Linear(len(ids) + 1, token_dim),
                                                 nn.GELU(), nn.LayerNorm(token_dim)))
        self.group_embedding = nn.Parameter(torch.randn(len(self.group_names), token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(token_dim, attention_heads, dim_feedforward=2 * token_dim,
                                            dropout=0.0, activation="gelu", batch_first=True,
                                            norm_first=True)
        self.intergroup = (nn.TransformerEncoder(layer, attention_layers, enable_nested_tensor=False)
                           if attention_layers else nn.Identity())
        self.pool_query = nn.Parameter(torch.randn(1, 1, token_dim) * .02)
        self.pool_attention = nn.MultiheadAttention(token_dim, attention_heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.GELU(),
                                 nn.LayerNorm(hidden_dim))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] != self.feature_dim:
            raise ValueError("Profile coordinate dimension mismatch")
        tokens = []
        for k, network in enumerate(self.group_nets):
            group = y.index_select(-1, getattr(self, f"indices_{k}"))
            finite = torch.isfinite(group)
            group = torch.where(finite, group, torch.zeros_like(group))
            availability = finite.to(group.dtype).mean(-1, keepdim=True)
            tokens.append(network(torch.cat((group, availability), -1)) + self.group_embedding[k])
        tokens = torch.stack(tokens, -2)
        leading = tokens.shape[:-2]
        tokens = tokens.reshape(-1, len(self.group_names), self.hidden_dim)
        if tokens.shape[0] == 0:
            return y.new_empty((*leading, self.hidden_dim))
        tokens = self.intergroup(tokens)
        pooled, _ = self.pool_attention(self.pool_query.expand(len(tokens), -1, -1),
                                       tokens, tokens, need_weights=False)
        return self.out(pooled.squeeze(-2)).reshape(*leading, self.hidden_dim)


def encode_library_profiles(encoder,condition_encoder,y,cond,mask,index=None):
    """Exact computational deduplication of repeated physical library X rows.

    Every neighbor remains in attention. This only shares deterministic encoder
    computation when multiple queries retrieve the same fitted-bank well.
    """
    if index is None:
        return encoder(y)+condition_encoder(cond)
    if index.shape != mask.shape or index.dtype != torch.int64 or torch.any(mask & (index<0)):
        raise ValueError("Available library indices must be nonnegative int64 bank rows")
    valid=torch.nonzero(mask.reshape(-1),as_tuple=False).flatten()
    output=y.new_zeros((mask.numel(),encoder.hidden_dim))
    if not len(valid):
        return output.reshape(*mask.shape,encoder.hidden_dim)
    key=index.reshape(-1)[valid]
    unique,inverse=torch.unique(key,sorted=True,return_inverse=True)
    first=torch.full((len(unique),),mask.numel(),dtype=torch.int64,device=y.device)
    first=first.scatter_reduce(0,inverse,valid,reduce="amin",include_self=True)
    flat_y=y.reshape(-1,y.shape[-1]);flat_cond=cond.reshape(-1,cond.shape[-1])
    if not torch.allclose(flat_y[valid],flat_y[first[inverse]],rtol=0,atol=0,equal_nan=True):
        raise ValueError("One library bank row identifies conflicting profiles")
    if not torch.equal(flat_cond[valid],flat_cond[first[inverse]]):
        raise ValueError("One library bank row identifies conflicting conditions")
    encoded=encoder(flat_y[first])+condition_encoder(flat_cond[first])
    output=output.index_copy(0,valid,encoded[inverse])
    return output.reshape(*mask.shape,encoder.hidden_dim)


def library_global_features(batch, available, dtype):
    """Ignore unavailable global-library padding before any neural arithmetic."""
    count=batch["library_count"].to(dtype).reshape(-1,1)
    pieces=[batch["library_global_mean"],batch["library_global_variance"],batch["library_density"],count]
    pieces=[torch.where(available.unsqueeze(-1),p,torch.zeros_like(p)) for p in pieces]
    if any(not torch.isfinite(p).all() for p in pieces) or torch.any(pieces[-1]<0):
        raise ValueError("Available global library statistics must be finite with nonnegative count")
    pieces[-1]=torch.log1p(pieces[-1])
    return torch.cat(pieces,-1)


class HierarchicalReferenceEncoder(nn.Module):
    """Condition on available source, batch and plate reference summaries.

    A missing level has a learned missingness token, never a fabricated measured
    reference. The raw numerical zeros used for masked tensor arithmetic are not
    presented to the level encoder as if they were observations.
    """
    def __init__(self, reference_dim: int, hidden_dim: int):
        super().__init__()
        self.reference_dim = reference_dim
        self.level_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(reference_dim, hidden_dim), nn.GELU(),
                          nn.LayerNorm(hidden_dim)) for _ in range(3)])
        self.missing = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.combine = nn.Sequential(nn.Linear(3 * hidden_dim + 3, hidden_dim),
                                     nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, reference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if reference.shape[-2:] != (3, self.reference_dim) or mask.shape != reference.shape[:-1]:
            raise ValueError("Reference summaries/masks have incompatible dimensions")
        if mask.dtype != torch.bool:
            raise ValueError("Reference masks must be boolean")
        tokens = []
        for level, network in enumerate(self.level_nets):
            valid = mask[..., level]
            x = reference[..., level, :]
            if torch.any(valid & ~torch.isfinite(x).all(-1)):
                raise ValueError("An available reference contains non-finite values")
            x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
            encoded = network(x)
            tokens.append(torch.where(valid.unsqueeze(-1), encoded, self.missing[level]))
        return self.combine(torch.cat((*tokens, mask.to(reference.dtype)), -1))


@dataclass
class JointGaussian:
    """Joint distribution with per-compound marginal and shared environments.

    ``factors`` specifies each compound's marginal covariance. When hierarchical
    metadata are present, local_factors are independently drawn per compound,
    whereas environment_loadings use the same latent draw for every occurrence
    of a source/batch/plate across all compounds in this distribution. Thus
    log_prob returns marginal per-compound scores, not a factorization of the
    complete likelihood; joint_log_prob is the complete batch likelihood.
    """
    mean: torch.Tensor
    diag_var: torch.Tensor
    factors: torch.Tensor
    local_factors: torch.Tensor | None = None
    environment_loadings: tuple[torch.Tensor, ...] | None = None
    environment_groups: torch.Tensor | None = None
    latent_semantics: str = "declared_hierarchical_measurement_factors"
    environment_cache_namespace: object | None = None

    @property
    def marginal_variance(self) -> torch.Tensor:
        return self.diag_var + self.factors.square().sum(-1)

    def sample_joint(self, n_samples: int, generator: torch.Generator | None = None,
                     environment_noise_cache: EnvironmentNoiseCache | None = None) -> torch.Tensor:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        b, t, d = self.mean.shape
        independent = torch.randn(n_samples, b, t, d, device=self.mean.device,
                                  dtype=self.mean.dtype, generator=generator)
        value = self.mean.unsqueeze(0) + independent * self.diag_var.sqrt().unsqueeze(0)
        local = self.factors if self.local_factors is None else self.local_factors
        latent = torch.randn(n_samples, b, local.shape[-1], device=self.mean.device,
                             dtype=self.mean.dtype, generator=generator)
        value = value + torch.einsum("btdl,sbl->sbtd", local.to(self.mean.dtype), latent)
        for level, (load, group_index, n_groups) in enumerate(self._environment_specs()):
            if environment_noise_cache is None:
                shared = torch.randn(n_samples, n_groups, load.shape[-1], device=self.mean.device,
                                     dtype=self.mean.dtype, generator=generator)
            else:
                flat_index = group_index.reshape(-1)
                flat_group = self.environment_groups.reshape(-1, 3)
                draws = []
                for index in range(n_groups):
                    first = torch.nonzero(flat_index == index, as_tuple=False)[0, 0]
                    ids = tuple(int(x) for x in flat_group[first, :level + 1].tolist())
                    if any(x < 0 for x in ids):
                        draw = torch.randn(n_samples, load.shape[-1], device=self.mean.device,
                                            dtype=self.mean.dtype, generator=generator)
                    else:
                        key = (("source", "batch", "plate")[level], *ids)
                        if self.environment_cache_namespace is not None:
                            key = (self.environment_cache_namespace, *key)
                        draw = environment_noise_cache.draw(
                            key, n_samples, load.shape[-1], device=self.mean.device,
                            dtype=self.mean.dtype, generator=generator)
                    draws.append(draw)
                shared = torch.stack(draws, dim=1)
            value = value + torch.einsum("btdr,sbtr->sbtd", load.to(self.mean.dtype), shared[:, group_index, :])
        return value

    def _environment_specs(self) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        if self.environment_loadings is None:
            return []
        if self.environment_groups is None or self.local_factors is None:
            raise ValueError("Shared environments require group IDs and local factors")
        b, t, _ = self.mean.shape
        if self.environment_groups.shape != (b, t, 3) or len(self.environment_loadings) != 3:
            raise ValueError("Three hierarchical group levels are required")
        specs = []
        groups = self.environment_groups.reshape(b * t, 3)
        unique_unknown = torch.arange(1, b * t + 1, device=groups.device)
        for level, load in enumerate(self.environment_loadings):
            # Composite keys enforce nesting: a batch number reused at another
            # source is not the same batch. Unknown IDs never imply sharing.
            prefix = groups[:, :level + 1]
            known = (prefix >= 0).all(-1)
            unknown_key = torch.where(known, torch.zeros_like(unique_unknown), unique_unknown)
            keys = torch.cat((prefix, unknown_key.unsqueeze(-1)), -1)
            unique, indices = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
            specs.append((load, indices.reshape(b, t), unique.shape[0]))
        return specs

    @staticmethod
    def _global_factor_block(batch_index: int,
                             specs: list[tuple[torch.Tensor, torch.Tensor, int]]) -> torch.Tensor:
        """Construct only one compound's N_b by R_global block, never full N²."""
        parts = []
        for load, group_index, n_groups in specs:
            assignment = F.one_hot(group_index[batch_index], num_classes=n_groups).to(load.dtype)
            expanded = load[batch_index].unsqueeze(-2) * assignment[:, None, :, None]
            parts.append(expanded.reshape(-1, n_groups * load.shape[-1]))
        return torch.cat(parts, -1)

    def weighted_moments(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact moments of a weighted profile sum, including shared covariance.

        For weights [B,T] this returns mean [B,D], independent diagonal [B,D]
        and factor [B,D,L]. Its covariance is diag(diagonal)+factor@factor.T,
        precisely the feature-wise matrix version of w.T @ Sigma @ w. Weights
        need not sum to one: averages, differences and sums are all supported.
        No equal-correlation or independent-repeat approximation is made.
        """
        if weights.shape != self.mean.shape[:2] or not torch.isfinite(weights).all():
            raise ValueError("weights must be finite [B,T]")
        mean = (weights.unsqueeze(-1) * self.mean).sum(1)
        diagonal = (weights.square().unsqueeze(-1) * self.diag_var).sum(1)
        factor = (weights[:, :, None, None] * self.factors).sum(1)
        return mean, diagonal, factor

    def aggregate_wells(self, weights: torch.Tensor) -> "JointGaussian":
        mean, diagonal, factor = self.weighted_moments(weights)
        specs = self._environment_specs()
        if not specs:
            return JointGaussian(mean.unsqueeze(1), diagonal.unsqueeze(1), factor.unsqueeze(1))
        local = (weights[:, :, None, None] * self.local_factors).sum(1, keepdim=True)
        # Compile each hierarchy level into a common global latent basis before
        # combining wells; this preserves cross-compound environmental sharing.
        environmental = []
        for load, group_index, n_groups in specs:
            by_compound = []
            for b in range(self.mean.shape[0]):
                assignment = F.one_hot(group_index[b], num_classes=n_groups).to(load.dtype)
                expanded = load[b].unsqueeze(-2) * assignment[:, None, :, None]
                weighted = (expanded * weights[b, :, None, None, None]).sum(0)
                by_compound.append(weighted.reshape(self.mean.shape[-1], -1))
            environmental.append(torch.stack(by_compound).unsqueeze(1))
        marginal_factor = torch.cat((local, *environmental), -1)
        groups = torch.zeros((self.mean.shape[0], 1, 3), dtype=torch.int64, device=self.mean.device)
        return JointGaussian(mean.unsqueeze(1), diagonal.unsqueeze(1), marginal_factor,
                             local, tuple(environmental), groups,
                             "globally_aligned_aggregate_latent_basis", object())

    def condition(self, observed_y: torch.Tensor, observed_mask: torch.Tensor,
                  target_indices: Sequence[int], *, retain_observed: bool = False,
                  lazy: bool | None = None) -> "JointGaussian | LazyConditionalGaussian":
        """Exactly condition this one joint Gaussian on observations across B.

        This is not a neural-network re-encoding of a larger context. Its mean
        update uses observations from all compounds that share environmental
        factors. Kept future wells must not overlap observed coordinates. Their
        order follows target_indices. Missing observed coordinates are ignored.

        After conditioning, globally coupled posterior factors no longer have a
        unique source/batch/plate interpretation. They are represented as one
        universally shared posterior basis plus zero-loading auxiliary levels,
        preserving the existing sampling and exact joint-likelihood interfaces.
        """
        observed = self.observed_mask(observed_y, observed_mask)
        raw_indices=tuple(target_indices)
        if any(not isinstance(i,Integral) or isinstance(i,bool) for i in raw_indices):
            raise ValueError("target_indices must be integer well indices")
        indices = tuple(int(i) for i in raw_indices)
        if (not indices or len(set(indices)) != len(indices)
                or min(indices) < 0 or max(indices) >= self.mean.shape[1]):
            raise ValueError("target_indices must be distinct valid future well indices")
        index = torch.tensor(indices, device=self.mean.device, dtype=torch.long)
        if observed.index_select(1, index).any() and not retain_observed:
            raise ValueError("Retained future wells overlap observed coordinates")
        b, t, d = self.mean.shape
        future_t = len(indices)
        local_all = self.factors if self.local_factors is None else self.local_factors
        local_rank = local_all.shape[-1]
        specs = self._environment_specs()
        diagonal = torch.where(observed, self.diag_var, torch.ones_like(self.diag_var)).double()
        if torch.any(diagonal <= 0) or not torch.isfinite(diagonal).all():
            raise ValueError("Observed diagonal covariance must be finite and positive")
        residual = torch.where(observed, observed_y - self.mean, torch.zeros_like(self.mean)).double()
        local_observed = torch.where(observed.unsqueeze(-1), local_all,
                                      torch.zeros_like(local_all)).double()
        inverse_sqrt = diagonal.rsqrt()
        residual = residual * inverse_sqrt
        local_observed = local_observed * inverse_sqrt.unsqueeze(-1)
        flat_local = local_observed.reshape(b, t * d, local_rank)
        local_transpose = flat_local.transpose(-2, -1)
        a = local_transpose @ flat_local
        a = a + torch.eye(local_rank, device=self.mean.device, dtype=torch.float64).unsqueeze(0)
        chol_a = torch.linalg.cholesky(a)
        local_rhs = local_transpose @ residual.reshape(b, t * d, 1)
        local_mean = torch.cholesky_solve(local_rhs, chol_a)
        if specs:
            active = torch.cat([load for load, _, _ in specs], -1)
            active = torch.where(observed.unsqueeze(-1), active, torch.zeros_like(active)).double()
            active = active * inverse_sqrt.unsqueeze(-1)
            columns, offset = [], 0
            for load, group_index, n_groups in specs:
                rank = load.shape[-1]
                columns.append(offset + group_index.unsqueeze(-1) * rank
                               + torch.arange(rank, device=self.mean.device))
                offset += n_groups * rank
            columns = torch.cat(columns, -1)
            global_rank, active_rank = offset, active.shape[-1]
            well_cross = local_observed.transpose(-2, -1) @ active
            scatter_columns = columns[:, None].expand(-1, local_rank, -1, -1)
            cross = torch.zeros(b, local_rank, global_rank, device=self.mean.device, dtype=torch.float64)
            cross = cross.scatter_add(
                -1, scatter_columns.reshape(b, local_rank, t * active_rank),
                well_cross.permute(0, 2, 1, 3).reshape(b, local_rank, t * active_rank))
            solved_cross = torch.cholesky_solve(cross, chol_a)
            well_core = active.transpose(-2, -1) @ active
            pair_columns = columns.unsqueeze(-1) * global_rank + columns.unsqueeze(-2)
            k = torch.zeros(global_rank * global_rank, device=self.mean.device, dtype=torch.float64)
            k = k.scatter_add(0, pair_columns.reshape(-1), well_core.reshape(-1)).reshape(global_rank, global_rank)
            k = k - (cross.transpose(-2, -1) @ solved_cross).sum(0)
            k = k + torch.eye(global_rank, device=self.mean.device, dtype=torch.float64)
            well_rhs = (active.transpose(-2, -1) @ residual.unsqueeze(-1)).squeeze(-1)
            q = torch.zeros(global_rank, device=self.mean.device, dtype=torch.float64)
            q = q.scatter_add(0, columns.reshape(-1), well_rhs.reshape(-1)).unsqueeze(-1)
            q = q - (cross.transpose(-2, -1) @ local_mean).sum(0)
            chol_k = torch.linalg.cholesky((k + k.T) * 0.5)
            global_mean = torch.cholesky_solve(q, chol_k)
        estimated_elements=b*future_t*d*(local_rank + (global_rank if specs else 0))
        if lazy is True or (lazy is None and estimated_elements>20_000_000):
            return LazyConditionalGaussian(
                self,observed_y,observed,index,chol_a,local_mean.squeeze(-1),
                solved_cross if specs else None,chol_k if specs else None,
                global_mean.squeeze(-1) if specs else None,retain_observed)
        future_mean, future_local, future_global = [], [], []
        for i in range(b):
            lt = local_all[i].index_select(0, index).reshape(future_t * d, local_rank).double()
            mu = self.mean[i].index_select(0, index).reshape(-1).double()
            mu = mu + (lt @ local_mean[i]).squeeze(-1)
            local_factor = torch.linalg.solve_triangular(chol_a[i], lt.T, upper=False).T
            if specs:
                # One compound block at a time avoids a full B*T*D*R_global
                # allocation while compiling the globally coupled posterior.
                gt = self._global_factor_block(i, specs).reshape(t, d, -1).index_select(0, index)
                gt = gt.reshape(future_t * d, -1).double()
                effective = gt - lt @ solved_cross[i]
                mu = mu + (effective @ global_mean).squeeze(-1)
                global_factor = torch.linalg.solve_triangular(chol_k, effective.T, upper=False).T
                future_global.append(global_factor.reshape(future_t, d, -1).to(self.mean.dtype))
            future_mean.append(mu.reshape(future_t, d).to(self.mean.dtype))
            future_local.append(local_factor.reshape(future_t, d, local_rank).to(self.mean.dtype))
        mean = torch.stack(future_mean)
        local = torch.stack(future_local)
        variance = self.diag_var.index_select(1, index)
        retained_observed = observed.index_select(1, index)
        if retain_observed:
            # Observed coordinates are exact point masses. They can stay in a
            # rectangular candidate tensor while other compounds retain the
            # same not-yet-bought role. Sampling/utility support zero variance;
            # density scoring must mask these degenerate coordinates out.
            mean = torch.where(retained_observed, observed_y.index_select(1, index), mean)
            variance = torch.where(retained_observed, torch.zeros_like(variance), variance)
            local = local.masked_fill(retained_observed.unsqueeze(-1), 0)
        if not specs:
            return JointGaussian(mean, variance, local, latent_semantics="exact_local_gaussian_posterior")
        global_factor = torch.stack(future_global)
        if retain_observed:
            global_factor = global_factor.masked_fill(retained_observed.unsqueeze(-1), 0)
        zeros = torch.zeros(b, future_t, d, 1, device=self.mean.device, dtype=self.mean.dtype)
        groups = torch.zeros(b, future_t, 3, dtype=torch.int64, device=self.mean.device)
        marginal = torch.cat((local, global_factor), -1)
        return JointGaussian(mean, variance, marginal, local,
                             (global_factor, zeros, zeros), groups,
                             "exact_posterior_global_basis_not_biological_decomposition", object())

    def observed_mask(self, target_y: torch.Tensor,
                      target_mask: torch.Tensor | None = None) -> torch.Tensor:
        if target_y.shape != self.mean.shape:
            raise ValueError("Target measurement shape mismatch")
        observed = torch.isfinite(target_y)
        if target_mask is not None:
            if target_mask.dtype != torch.bool:
                raise ValueError("Target masks must be boolean")
            if target_mask.shape == target_y.shape[:-1]:
                target_mask = target_mask.unsqueeze(-1)
            elif target_mask.shape != target_y.shape:
                raise ValueError("Target mask must index wells or coordinates")
            observed = observed & target_mask
        return observed

    def log_prob(self, target_y: torch.Tensor,
                 target_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Exact observed-coordinate marginal log likelihood via Woodbury.

        Computation uses float64 even when neural parameters use float32. Missing
        target coordinates are marginalized, not imputed into the likelihood.
        """
        observed = self.observed_mask(target_y, target_mask)
        results = []
        for i in range(self.mean.shape[0]):
            mask = observed[i].reshape(-1)
            if not mask.any():
                results.append(self.mean[i].sum() * 0)
                continue
            residual = (target_y[i].reshape(-1)[mask] - self.mean[i].reshape(-1)[mask]).double()
            diagonal = self.diag_var[i].reshape(-1)[mask].double()
            factor = self.factors[i].reshape(-1, self.factors.shape[-1])[mask].double()
            if torch.any(diagonal <= 0) or not torch.isfinite(diagonal).all():
                raise ValueError("Diagonal covariance must be finite and strictly positive")
            scaled_factor = factor / diagonal.sqrt().unsqueeze(-1)
            core = torch.eye(factor.shape[-1], device=factor.device, dtype=torch.float64)
            core = core + scaled_factor.T @ scaled_factor
            cholesky = torch.linalg.cholesky(core)
            b = factor.T @ (residual / diagonal)
            solved = torch.cholesky_solve(b.unsqueeze(-1), cholesky).squeeze(-1)
            quadratic = (residual.square() / diagonal).sum() - (b * solved).sum()
            logdet = diagonal.log().sum() + 2 * cholesky.diagonal().log().sum()
            results.append(-0.5 * (residual.numel() * math.log(2 * math.pi) + logdet + quadratic))
        return torch.stack(results)

    def joint_log_prob(self, target_y: torch.Tensor,
                       target_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Exact joint likelihood using sparse environmental sufficient statistics.

        Only the source/batch/plate factors active in each well are materialized
        in profile space. Their small cross-products are scattered into global
        latent coordinates before the same two-level Woodbury integration.
        """
        specs = self._environment_specs()
        if not specs:
            return self.log_prob(target_y, target_mask).sum()
        return self._joint_log_prob_sparse(target_y, target_mask, specs)

    def _joint_log_prob_sparse(self, target_y: torch.Tensor,
                               target_mask: torch.Tensor | None = None,
                               specs: list | None = None) -> torch.Tensor:
        """Exact sparse-in-global-columns, dense-in-local-rank Woodbury algebra."""
        specs = self._environment_specs() if specs is None else specs
        if not specs:
            return self.log_prob(target_y, target_mask).sum()
        observed = self.observed_mask(target_y, target_mask)
        if not observed.any():
            return self.mean.sum().double() * 0
        b, t, d = self.mean.shape
        diagonal = torch.where(observed, self.diag_var, torch.ones_like(self.diag_var)).double()
        if torch.any(diagonal <= 0) or not torch.isfinite(diagonal).all():
            raise ValueError("Diagonal covariance must be finite and positive")
        residual = torch.where(observed, target_y - self.mean, torch.zeros_like(self.mean)).double()
        local = torch.where(observed.unsqueeze(-1), self.local_factors,
                             torch.zeros_like(self.local_factors)).double()
        active = torch.cat([load for load, _, _ in specs], -1)
        active = torch.where(observed.unsqueeze(-1), active, torch.zeros_like(active)).double()
        columns, offset = [], 0
        for load, group_index, n_groups in specs:
            rank = load.shape[-1]
            columns.append(offset + group_index.unsqueeze(-1) * rank
                           + torch.arange(rank, device=group_index.device))
            offset += n_groups * rank
        columns = torch.cat(columns, -1)
        # Integrating environmental latents that touch only missing coordinates
        # contributes exactly one to this marginal density. Padding must not
        # add thousands of identity columns to its Cholesky factorization.
        # Keep every column belonging to an observed well, including unknown
        # groups and currently zero-valued loadings; this is mask-based only.
        observed_columns = torch.unique(columns[observed.any(-1)])
        column_map = torch.zeros(offset, dtype=torch.long, device=columns.device)
        column_map[observed_columns] = torch.arange(observed_columns.numel(), device=columns.device)
        columns = column_map[columns]
        # Wholly unobserved wells map safely to column zero: their residual and
        # all factor entries have already been set to zero above.
        global_rank, active_rank = observed_columns.numel(), active.shape[-1]
        inverse_sqrt = diagonal.rsqrt()
        r = residual * inverse_sqrt
        local = local * inverse_sqrt.unsqueeze(-1)
        active = active * inverse_sqrt.unsqueeze(-1)
        local_flat = local.reshape(b, t * d, -1)
        local_transpose = local_flat.transpose(-2, -1)
        local_core = local_transpose @ local_flat
        local_core = local_core + torch.eye(local.shape[-1], device=local.device,
                                            dtype=torch.float64).unsqueeze(0)
        local_cholesky = torch.linalg.cholesky(local_core)
        cross_residual = local_transpose @ r.reshape(b, t * d, 1)
        # Each well has only sum(level ranks) active environmental columns.
        # No D by global_rank tensor is formed.
        well_local_cross = local.transpose(-2, -1) @ active
        scatter_columns = columns[:, None, :, :].expand(-1, local.shape[-1], -1, -1)
        cross_global = torch.zeros(b, local.shape[-1], global_rank,
                                     device=local.device, dtype=torch.float64)
        cross_global = cross_global.scatter_add(
            -1, scatter_columns.reshape(b, local.shape[-1], t * active_rank),
            well_local_cross.permute(0, 2, 1, 3).reshape(b, local.shape[-1], t * active_rank))
        well_global_core = active.transpose(-2, -1) @ active
        pair_columns = columns.unsqueeze(-1) * global_rank + columns.unsqueeze(-2)
        diagonal_global_core = torch.zeros(global_rank * global_rank,
                                             device=local.device, dtype=torch.float64)
        diagonal_global_core = diagonal_global_core.scatter_add(
            0, pair_columns.reshape(-1), well_global_core.reshape(-1)).reshape(global_rank, global_rank)
        well_global_residual = (active.transpose(-2, -1) @ r.unsqueeze(-1)).squeeze(-1)
        diagonal_global_rhs = torch.zeros(global_rank, device=local.device, dtype=torch.float64)
        diagonal_global_rhs = diagonal_global_rhs.scatter_add(
            0, columns.reshape(-1), well_global_residual.reshape(-1)).unsqueeze(-1)
        solved_residual = torch.cholesky_solve(cross_residual, local_cholesky)
        solved_global = torch.cholesky_solve(cross_global, local_cholesky)
        global_core = diagonal_global_core - (cross_global.transpose(-2, -1) @ solved_global).sum(0)
        global_core = global_core + torch.eye(global_rank, device=local.device, dtype=torch.float64)
        global_rhs = diagonal_global_rhs - (cross_global.transpose(-2, -1) @ solved_residual).sum(0)
        global_core = (global_core + global_core.T) * 0.5
        global_cholesky = torch.linalg.cholesky(global_core)
        global_solution = torch.cholesky_solve(global_rhs, global_cholesky)
        quadratic = (r.square().sum() - (cross_residual * solved_residual).sum()
                     - (global_rhs * global_solution).sum())
        logdet = (diagonal.log().sum() + 2 * local_cholesky.diagonal(dim1=-2, dim2=-1).log().sum()
                  + 2 * global_cholesky.diagonal().log().sum())
        return -0.5 * (observed.sum().double() * math.log(2 * math.pi) + logdet + quadratic)

    def _joint_log_prob_batched(self, target_y: torch.Tensor,
                                target_mask: torch.Tensor | None = None,
                                specs: list | None = None) -> torch.Tensor:
        """Whitened batched sufficient-statistic form of two-level Woodbury."""
        specs = self._environment_specs() if specs is None else specs
        if not specs:
            return self.log_prob(target_y, target_mask).sum()
        observed = self.observed_mask(target_y, target_mask)
        if not observed.any():
            return self.mean.sum().double() * 0
        b = self.mean.shape[0]
        n = self.mean.shape[1] * self.mean.shape[2]
        diagonal = torch.where(observed, self.diag_var, torch.ones_like(self.diag_var)).reshape(b, n).double()
        if torch.any(diagonal <= 0) or not torch.isfinite(diagonal).all():
            raise ValueError("Diagonal covariance must be finite and positive")
        residual = torch.where(observed, target_y - self.mean, torch.zeros_like(self.mean)).reshape(b, n).double()
        local = torch.where(observed.unsqueeze(-1), self.local_factors,
                             torch.zeros_like(self.local_factors)).reshape(b, n, -1).double()
        parts = []
        for load, group_index, n_groups in specs:
            assignment = F.one_hot(group_index, num_classes=n_groups).to(load.dtype)
            expanded = load.unsqueeze(-2) * assignment[:, :, None, :, None]
            parts.append(expanded.reshape(b, n, n_groups * load.shape[-1]))
        global_factor = torch.cat(parts, -1)
        global_factor = torch.where(observed.reshape(b, n, 1), global_factor,
                                     torch.zeros_like(global_factor)).double()
        inverse_sqrt = diagonal.rsqrt()
        r = residual * inverse_sqrt
        local = local * inverse_sqrt.unsqueeze(-1)
        global_factor = global_factor * inverse_sqrt.unsqueeze(-1)
        local_transpose = local.transpose(-2, -1)
        local_core = local_transpose @ local
        local_core = local_core + torch.eye(local.shape[-1], device=local.device,
                                            dtype=torch.float64).unsqueeze(0)
        local_cholesky = torch.linalg.cholesky(local_core)
        cross_residual = local_transpose @ r.unsqueeze(-1)
        cross_global = local_transpose @ global_factor
        solved_residual = torch.cholesky_solve(cross_residual, local_cholesky)
        solved_global = torch.cholesky_solve(cross_global, local_cholesky)
        # Work entirely with small cross-products after whitening. This avoids
        # constructing A_b^-1 G_b as an additional N_b by R_global tensor.
        global_transpose = global_factor.transpose(-2, -1)
        global_core = (global_transpose @ global_factor
                       - cross_global.transpose(-2, -1) @ solved_global).sum(0)
        global_core = global_core + torch.eye(global_factor.shape[-1], device=local.device,
                                              dtype=torch.float64)
        global_rhs = (global_transpose @ r.unsqueeze(-1)
                      - cross_global.transpose(-2, -1) @ solved_residual).sum(0)
        global_core = (global_core + global_core.T) * 0.5
        global_cholesky = torch.linalg.cholesky(global_core)
        global_solution = torch.cholesky_solve(global_rhs, global_cholesky)
        quadratic = (r.square().sum() - (cross_residual * solved_residual).sum()
                     - (global_rhs * global_solution).sum())
        logdet = (diagonal.log().sum() + 2 * local_cholesky.diagonal(dim1=-2, dim2=-1).log().sum()
                  + 2 * global_cholesky.diagonal().log().sum())
        return -0.5 * (observed.sum().double() * math.log(2 * math.pi) + logdet + quadratic)

    def _joint_log_prob_blockwise(self, target_y: torch.Tensor,
                                  target_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Exact cross-compound likelihood via two-level Woodbury identities.

        First integrate compound/per-well local factors inside each block. Then
        integrate global source/batch/plate factors shared by blocks. Missing
        target coordinates are marginalized. Complexity depends on local/global
        latent ranks, not a dense (B*T*D)-square covariance matrix.
        """
        specs = self._environment_specs()
        if not specs:
            return self.log_prob(target_y, target_mask).sum()
        observed = self.observed_mask(target_y, target_mask)
        global_rank = sum(load.shape[-1] * n for load, _, n in specs)
        device = self.mean.device
        global_core = torch.eye(global_rank, device=device, dtype=torch.float64)
        global_rhs = torch.zeros(global_rank, device=device, dtype=torch.float64)
        total_logdet = torch.zeros((), device=device, dtype=torch.float64)
        total_quadratic = torch.zeros((), device=device, dtype=torch.float64)
        observed_count = 0
        for i in range(self.mean.shape[0]):
            mask = observed[i].reshape(-1)
            if not mask.any():
                continue
            residual = (target_y[i].reshape(-1)[mask] - self.mean[i].reshape(-1)[mask]).double()
            diagonal = self.diag_var[i].reshape(-1)[mask].double()
            if torch.any(diagonal <= 0) or not torch.isfinite(diagonal).all():
                raise ValueError("Diagonal covariance must be finite and positive")
            local = self.local_factors[i].reshape(-1, self.local_factors.shape[-1])[mask].double()
            global_factor = self._global_factor_block(i, specs)[mask].double()
            local_scaled = local / diagonal.sqrt().unsqueeze(-1)
            local_core = torch.eye(local.shape[-1], device=device, dtype=torch.float64)
            local_core = local_core + local_scaled.T @ local_scaled
            local_cholesky = torch.linalg.cholesky(local_core)
            right = torch.cat((residual.unsqueeze(-1), global_factor), -1)
            diagonal_solved = right / diagonal.unsqueeze(-1)
            inner = local.T @ diagonal_solved
            local_solved = torch.cholesky_solve(inner, local_cholesky)
            inverse_right = diagonal_solved - (local / diagonal.unsqueeze(-1)) @ local_solved
            inverse_residual, inverse_global = inverse_right[:, 0], inverse_right[:, 1:]
            global_core = global_core + global_factor.T @ inverse_global
            global_rhs = global_rhs + global_factor.T @ inverse_residual
            total_logdet = total_logdet + diagonal.log().sum() + 2 * local_cholesky.diagonal().log().sum()
            total_quadratic = total_quadratic + (residual * inverse_residual).sum()
            observed_count += residual.numel()
        if observed_count == 0:
            return self.mean.sum().double() * 0
        # Symmetrize away matrix-product roundoff before the small Cholesky.
        global_core = (global_core + global_core.T) * 0.5
        global_cholesky = torch.linalg.cholesky(global_core)
        solved = torch.cholesky_solve(global_rhs.unsqueeze(-1), global_cholesky).squeeze(-1)
        quadratic = total_quadratic - (global_rhs * solved).sum()
        logdet = total_logdet + 2 * global_cholesky.diagonal().log().sum()
        return -0.5 * (observed_count * math.log(2 * math.pi) + logdet + quadratic)




class LazyConditionalGaussian:
    """Exact conditional Gaussian without a dense future-by-global-rank basis.

    Let g be global environmental latents and u_i local latents. Woodbury gives
    g|observations ~ N(m_g,K^-1) and u_i|g,observations ~
    N(A_i^-1 h_i-A_i^-1 C_i g,A_i^-1). Sampling these blocks, then applying the
    ORIGINAL sparse loadings, is exactly the same Gaussian as materializing
    the posterior factors. No rank truncation or independent-hole shortcut is
    used. Further observations are merged into the original evidence, so rank
    does not grow after every planning step.
    """
    def __init__(self,parent,observed_y,observed,index,chol_a,local_mean,
                 solved_cross,chol_k,global_mean,retain_observed):
        self.parent=parent
        self.observed_y=observed_y
        self.observed=observed
        self.index=index
        self.chol_a=chol_a
        self.local_mean=local_mean
        self.solved_cross=solved_cross
        self.chol_k=chol_k
        self.global_mean=global_mean
        self.retain_observed=retain_observed
        self.cache_namespace=object()
        self.latent_semantics="exact_lazy_conditional_global_local_blocks_not_causal_decomposition"
        self.specs=parent._environment_specs()
        local=parent.factors if parent.local_factors is None else parent.local_factors
        self.local=local.index_select(1,index)
        self.future_observed=observed.index_select(1,index)
        center=local_mean
        if self.specs:
            center=center-torch.einsum("blr,r->bl",solved_cross,global_mean)
        mean=parent.mean.index_select(1,index).double()+torch.einsum("btdl,bl->btd",self.local.double(),center)
        if self.specs:
            offset=0
            for load,group,n_groups in self.specs:
                rank=load.shape[-1]
                columns=offset+group.index_select(1,index).unsqueeze(-1)*rank+torch.arange(rank,device=mean.device)
                mean=mean+torch.einsum("btdr,btr->btd",load.index_select(1,index).double(),global_mean[columns])
                offset+=n_groups*rank
        self.mean=mean.to(parent.mean.dtype)
        self.diag_var=parent.diag_var.index_select(1,index)
        if retain_observed:
            self.mean=torch.where(self.future_observed,observed_y.index_select(1,index),self.mean)
            self.diag_var=self.diag_var.masked_fill(self.future_observed,0)

    def sample_joint(self,n_samples,generator=None,environment_noise_cache=None):
        if n_samples<1:
            raise ValueError("n_samples must be positive")
        b,t,d=self.mean.shape
        rank=self.local.shape[-1]
        device,dtype=self.mean.device,self.mean.dtype
        local_noise=torch.randn(b,rank,n_samples,device=device,dtype=torch.float64,generator=generator)
        local_draw=torch.linalg.solve_triangular(self.chol_a.transpose(-2,-1),local_noise,upper=True)
        local_draw=local_draw.permute(2,0,1)+self.local_mean.unsqueeze(0)
        global_draw=None
        if self.specs:
            r=len(self.global_mean)
            if environment_noise_cache is None:
                noise=torch.randn(n_samples,r,device=device,dtype=torch.float64,generator=generator)
            else:
                noise=environment_noise_cache.draw(("conditional_global",self.cache_namespace),n_samples,r,
                         device=device,dtype=torch.float64,generator=generator)
            global_draw=torch.linalg.solve_triangular(self.chol_k.T,noise.T,upper=True).T+self.global_mean
            local_draw=local_draw-torch.einsum("blr,sr->sbl",self.solved_cross,global_draw)
        draw=self.parent.mean.index_select(1,self.index).unsqueeze(0)
        draw=draw+torch.einsum("btdl,sbl->sbtd",self.local.to(dtype),local_draw.to(dtype))
        if self.specs:
            offset=0
            for load,group,n_groups in self.specs:
                r=load.shape[-1]
                columns=offset+group.index_select(1,self.index).unsqueeze(-1)*r+torch.arange(r,device=device)
                draw=draw+torch.einsum("btdr,sbtr->sbtd",load.index_select(1,self.index).to(dtype),global_draw[:,columns].to(dtype))
                offset+=n_groups*r
        independent=torch.randn(n_samples,b,t,d,device=device,dtype=dtype,generator=generator)
        draw=draw+independent*self.parent.diag_var.index_select(1,self.index).sqrt().unsqueeze(0)
        if self.retain_observed:
            draw=torch.where(self.future_observed.unsqueeze(0),self.observed_y.index_select(1,self.index).unsqueeze(0),draw)
        return draw

    def _new_observations(self,y,mask=None):
        if y.shape!=self.mean.shape:
            raise ValueError("Conditional target shape mismatch")
        measured=torch.isfinite(y)
        if mask is not None:
            if mask.dtype!=torch.bool:
                raise ValueError("Conditional masks must be boolean")
            if mask.shape==y.shape[:-1]: mask=mask.unsqueeze(-1)
            if mask.shape!=y.shape and mask.shape!=(*y.shape[:-1],1):
                raise ValueError("Conditional mask shape mismatch")
            measured=measured&mask
        overlap=measured&self.future_observed
        old=self.observed_y.index_select(1,self.index)
        if torch.any(overlap & (y!=old)):
            raise ValueError("New values conflict with already observed point masses")
        full_y=self.observed_y.clone()
        full_mask=self.observed.clone()
        old_y=full_y.index_select(1,self.index)
        full_y=full_y.index_copy(1,self.index,torch.where(measured,y,old_y))
        full_mask=full_mask.index_copy(1,self.index,full_mask.index_select(1,self.index)|measured)
        return full_y,full_mask

    def condition(self,observed_y,observed_mask,target_indices,*,retain_observed=False,lazy=True):
        indices=tuple(target_indices)
        if (not indices or any(not isinstance(i,Integral) or isinstance(i,bool) for i in indices)
                or len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=len(self.index)):
            raise ValueError("target_indices must be distinct integer conditional roles")
        full_y,full_mask=self._new_observations(observed_y,observed_mask)
        parent_indices=self.index[list(indices)].tolist()
        return self.parent.condition(full_y,full_mask,parent_indices,retain_observed=retain_observed,lazy=lazy)

    def joint_log_prob(self,target_y,target_mask=None):
        full_y,full_mask=self._new_observations(target_y,target_mask)
        return self.parent.joint_log_prob(full_y,full_mask)-self.parent.joint_log_prob(self.observed_y,self.observed)


class StructuredFactorHead(nn.Module):
    """Trainable D by rank basis with state-dependent amplitudes/group gates.

    Rank is not reduced: the complete requested covariance basis is learned.
    Factorizing the parameter map avoids H*D*rank coefficients per hierarchy.
    """
    def __init__(self, hidden_dim, feature_group_index, rank):
        super().__init__()
        self.rank = rank
        self.register_buffer("feature_group_index", feature_group_index.clone())
        self.weight = nn.Parameter(torch.randn(len(feature_group_index), rank) * .08)
        self.bias = nn.Parameter(torch.zeros(len(feature_group_index), rank))
        self.amplitude = nn.Linear(hidden_dim, rank)
        self.group_gate = nn.Linear(hidden_dim, int(feature_group_index.max()) + 1)
        nn.init.zeros_(self.amplitude.bias)
        nn.init.zeros_(self.group_gate.bias)

    def forward(self, state):
        amplitude = F.softplus(self.amplitude(state)) + .1
        gate = torch.exp(.5 * torch.tanh(self.group_gate(state)))
        gate = gate.index_select(-1, self.feature_group_index)
        return self.weight * amplitude.unsqueeze(-2) * gate.unsqueeze(-1) + self.bias


class GroupHeteroscedasticScale(nn.Module):
    """Feature baseline times compound/global and compartment-family scales."""
    def __init__(self, hidden_dim, feature_group_index, min_scale):
        super().__init__()
        self.register_buffer("feature_group_index", feature_group_index.clone())
        self.weight = nn.Parameter(torch.empty(int(feature_group_index.max()) + 1, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(int(feature_group_index.max()) + 1))
        nn.init.xavier_uniform_(self.weight, gain=.2)
        self.compound_scale = nn.Linear(hidden_dim, 1)
        self.feature_log_scale = nn.Parameter(torch.zeros(len(feature_group_index)))
        self.min_scale = min_scale

    def forward(self, state):
        group = F.linear(state, self.weight, self.bias).index_select(-1, self.feature_group_index)
        log_scale = self.feature_log_scale + group + self.compound_scale(state)
        return F.softplus(log_scale) + self.min_scale


class IdentityMatchedReferenceEncoder(nn.Module):
    """Real matched controls, hierarchical pooling, and leave-one-control-out reconstruction.

    Templates must be estimated outside this module from fitting-only controls.
    No Target-2 profile is inferred from a DMSO summary. A missing template is
    not an identity match; it can inform raw-reference encoding but not affine
    anchoring or the identity-matched reconstruction objective.
    """
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.token = nn.Sequential(nn.Linear(3 * feature_dim + 1, hidden_dim), nn.GELU(),
                                   nn.LayerNorm(hidden_dim))
        self.level_embedding = nn.Parameter(torch.randn(3, hidden_dim) * .02)
        self.missing = nn.Parameter(torch.randn(3, hidden_dim) * .02)
        self.combine = nn.Sequential(nn.Linear(3 * hidden_dim + 3, hidden_dim), nn.GELU(),
                                     nn.LayerNorm(hidden_dim))
        self.template_encoder = nn.Linear(feature_dim, hidden_dim)
        self.reconstruct = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
                                         nn.Linear(hidden_dim, feature_dim))

    def forward(self, y, template, mask, template_mask, identity=None):
        if y.shape != template.shape or y.shape[:-1] != mask.shape or mask.dtype != torch.bool:
            raise ValueError("Reference panel arrays/masks are incompatible")
        if template_mask.shape != mask.shape or template_mask.dtype != torch.bool:
            raise ValueError("Reference template masks must match controls")
        if y.shape[-3] != 3 or y.shape[-1] != self.feature_dim:
            raise ValueError("Panels must have three levels and fixed measurement coordinates")
        paired = mask & template_mask
        yy = torch.where(mask.unsqueeze(-1), y, torch.zeros_like(y))
        tt = torch.where(paired.unsqueeze(-1), template, torch.zeros_like(template))
        if not torch.isfinite(yy).all() or not torch.isfinite(tt).all():
            raise ValueError("Available reference profiles/templates must be finite")
        token = self.token(torch.cat((yy, tt, (yy - tt) * paired.unsqueeze(-1),
                                      paired.unsqueeze(-1).to(y.dtype)), -1))
        token = token + self.level_embedding[:, None, :]
        count = mask.sum(-1)
        summed = (token * mask.unsqueeze(-1)).sum(-2)
        pooled = summed / count.clamp_min(1).unsqueeze(-1)
        pooled = torch.where((count > 0).unsqueeze(-1), pooled, self.missing)
        encoded = self.combine(torch.cat((pooled.flatten(-2), (count > 0).to(y.dtype)), -1))
        # A known template of this identity plus OTHER controls predicts each
        # reference. Its own measurement does not enter its reconstruction.
        # The templates of OTHER controls may have been fitted using the held
        # control. Therefore the reconstruction's other-control path uses raw
        # measurements only, never those other templates. The held control's
        # own template is separately fitted leaving that physical control out.
        # This closes a subtle indirect self-reconstruction shortcut.
        raw_token=self.token(torch.cat((yy,torch.zeros_like(yy),torch.zeros_like(yy),
                                       torch.zeros_like(paired.unsqueeze(-1),dtype=y.dtype)),-1))
        raw_token=raw_token+self.level_embedding[:,None,:]
        raw_sum=(raw_token*mask.unsqueeze(-1)).sum(-2)
        other = (raw_sum.unsqueeze(-2) - raw_token * mask.unsqueeze(-1))
        other = other / (count.unsqueeze(-1) - mask.to(count.dtype)).clamp_min(1).unsqueeze(-1)
        reconstructed = tt + self.reconstruct(torch.cat((other, self.template_encoder(tt)), -1))
        reconstruction_mask = paired & (count.unsqueeze(-1) >= 2)
        error = (reconstructed - yy).square().mean(-1)
        reconstruction = (error * reconstruction_mask).sum() / reconstruction_mask.sum().clamp_min(1)
        # Ridge affine fit for each coordinate from matched controls. A single
        # identity/control supplies offset but cannot identify arbitrary scale:
        # the ridge shrinks that slope exactly toward 1.
        pair_count = paired.sum(-1, keepdim=True).clamp_min(1)
        my = (yy * paired.unsqueeze(-1)).sum(-2) / pair_count
        mt = (tt * paired.unsqueeze(-1)).sum(-2) / pair_count
        yc, tc = (yy - my.unsqueeze(-2)), (tt - mt.unsqueeze(-2))
        cov = (yc * tc * paired.unsqueeze(-1)).sum(-2)
        var = (tc.square() * paired.unsqueeze(-1)).sum(-2)
        slope = (cov + 1.) / (var + 1.)
        # Replicated DMSO templates (especially leave-own-control-out means)
        # do not supply multiple reference effects. Their numerical variation
        # must not masquerade as identifying an affine slope. Scale adaptation
        # requires at least two explicitly distinct matched identities.
        sufficient_identity = torch.zeros_like(paired.any(-1))
        if identity is not None:
            if identity.shape != mask.shape or identity.dtype != torch.int64:
                raise ValueError("Panel reference identities must be int64 and aligned")
            known = paired & (identity >= 0)
            p=identity.shape[-1]
            earlier=torch.arange(p,device=y.device)[None,:] < torch.arange(p,device=y.device)[:,None]
            same=(identity.unsqueeze(-1)==identity.unsqueeze(-2)) & known.unsqueeze(-2) & earlier
            distinct=(known & ~same.any(-1)).sum(-1)
            sufficient_identity=distinct>=2
        slope=torch.where(sufficient_identity.unsqueeze(-1),slope,torch.ones_like(slope))
        # Robust positivity is an explicit affine measurement assumption, not
        # clipping of the scientific utility or target outcome.
        slope = slope.clamp(.05, 20.)
        offset = my - slope * mt
        available = paired.any(-1)
        # Most local available level wins; source -> batch -> plate.
        choice = torch.where(available, torch.arange(3, device=y.device), -1).max(-1).values
        onehot = F.one_hot(choice.clamp_min(0), 3).to(y.dtype) * (choice >= 0).unsqueeze(-1)
        affine_scale = (slope * onehot.unsqueeze(-1)).sum(-2)
        affine_scale = torch.where((choice >= 0).unsqueeze(-1), affine_scale, torch.ones_like(affine_scale))
        affine_offset = (offset * onehot.unsqueeze(-1)).sum(-2)
        return encoded, reconstruction, affine_scale, affine_offset, reconstruction_mask.sum()

    def forward_compact(self, catalog_y, catalog_template, catalog_template_mask, index, mask, identity=None):
        """Encode each distinct physical panel once, without B*W*P*D expansion."""
        if index.dtype != torch.int64 or index.shape != mask.shape or mask.dtype != torch.bool:
            raise ValueError("Compact reference indices/masks must be int64/bool and aligned")
        if torch.any(mask & ((index < 0) | (index >= len(catalog_y)))):
            raise ValueError("An available reference index is outside its accessible catalog")
        leading, p = index.shape[:-2], index.shape[-1]
        normalized = torch.where(mask, index, -1)
        flattened = normalized.reshape(-1, 3 * p)
        unique, inverse = torch.unique(flattened, dim=0, return_inverse=True)
        outputs, losses, scales, offsets, counts = [], [], [], [], []
        for unique_index,row in enumerate(unique):
            indices = row.reshape(1,3,p)
            valid = indices >= 0
            safe = indices.clamp_min(0)
            yy = catalog_y[safe]
            tt = catalog_template[safe]
            matched = catalog_template_mask[safe] & valid
            ids=None
            if identity is not None:
                first=torch.nonzero(inverse==unique_index,as_tuple=False)[0,0]
                ids=identity.reshape(-1,3,p)[first].unsqueeze(0)
            encoded, loss, scale, offset, count = self(yy,tt,valid,matched,ids)
            outputs.append(encoded[0]); losses.append(loss); scales.append(scale[0]); offsets.append(offset[0]); counts.append(count)
        if not outputs:
            z=catalog_y.sum()*0
            return (catalog_y.new_empty((*leading,self.hidden_dim)),z,
                    catalog_y.new_empty((*leading,self.feature_dim)),catalog_y.new_empty((*leading,self.feature_dim)),
                    torch.zeros((),dtype=torch.long,device=catalog_y.device))
        # Each physical reference panel contributes once to the auxiliary loss,
        # not once per compound that happens to point at it in this minibatch.
        counts = torch.stack(counts)
        loss=(torch.stack(losses)*counts).sum()/counts.sum().clamp_min(1)
        return (torch.stack(outputs)[inverse].reshape(*leading,self.hidden_dim),loss,
                torch.stack(scales)[inverse].reshape(*leading,self.feature_dim),
                torch.stack(offsets)[inverse].reshape(*leading,self.feature_dim),counts.sum())


class ChemicalGaussianPrior(nn.Module):
    """A real diagonal Gaussian p(z | chemistry), with an explicit missing prior."""
    def __init__(self, chemical_dim, hidden_dim, rank):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(chemical_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.mean = nn.Linear(hidden_dim, rank)
        self.log_variance = nn.Linear(hidden_dim, rank)
        self.missing_token = nn.Parameter(torch.zeros(hidden_dim))
        self.missing_mean = nn.Parameter(torch.zeros(rank))
        self.missing_log_variance = nn.Parameter(torch.zeros(rank))

    def forward(self, chem, mask=None):
        if mask is None:
            mask = torch.ones(chem.shape[0], dtype=torch.bool, device=chem.device)
        if mask.shape != chem.shape[:1] or mask.dtype != torch.bool:
            raise ValueError("chem_mask must be boolean [B]")
        clean = torch.where(mask.unsqueeze(-1), chem, torch.zeros_like(chem))
        if not torch.isfinite(clean).all():
            raise ValueError("Available chemical descriptors must be finite")
        encoded = self.network(clean)
        token = torch.where(mask.unsqueeze(-1), encoded, self.missing_token)
        mean = torch.where(mask.unsqueeze(-1), self.mean(encoded), self.missing_mean)
        logvar = torch.where(mask.unsqueeze(-1), self.log_variance(encoded), self.missing_log_variance)
        return token, mean, torch.exp(logvar.clamp(-10, 10))


class MeasurementWorldModel(nn.Module):
    """R2 conditional latent neural process in fixed measurement coordinates.

    p(z|chem) is Gaussian; amortized Gaussian context factors define p(z|S).
    Training uses q(z|S,Y_target) only inside a conditional variational bound.
    This is not a claim that encoded evidence is the exact posterior of raw
    measurements. The Gaussian decoder is linear in z, so deployed p(Y|S) is
    marginalized exactly and all probe updates use that same joint Gaussian.
    """
    def __init__(self, feature_groups, condition_dim, reference_dim, chemical_dim,
                 hidden_dim=256, latent_rank=32, residual_rank=8, kernel_mode="measurement",
                 min_scale=1e-3, dropout=0., identity_transport=True,
                 group_attention_layers=2, attention_heads=4, reference_loss_weight=.1,
                 latent_kl_weight=1., chemical_regularization_weight=0.,
                 use_chemistry=True, use_references=True, use_library=True,
                 use_biology_prior=False, biology_vocabulary=None,
                 biology_kernel_mode="off", chemical_anchor_data=None,
                 observation_family="gaussian"):
        nn.Module.__init__(self)
        if min(condition_dim, reference_dim, chemical_dim, hidden_dim, latent_rank, residual_rank) < 1:
            raise ValueError("Dimensions and ranks must be positive")
        if min_scale <= 0 or min(reference_loss_weight, latent_kl_weight, chemical_regularization_weight) < 0:
            raise ValueError("Invalid positive scale or loss weights")
        if latent_kl_weight != 1.:
            raise ValueError("A conditional ELBO requires latent_kl_weight=1; beta objectives must be separately specified")
        self.config = dict(feature_groups={k:list(v) for k,v in feature_groups.items()},
                           condition_dim=condition_dim, reference_dim=reference_dim, chemical_dim=chemical_dim,
                           hidden_dim=hidden_dim, latent_rank=latent_rank, residual_rank=residual_rank,
                           kernel_mode=kernel_mode, min_scale=min_scale, dropout=dropout,
                           identity_transport=identity_transport, group_attention_layers=group_attention_layers,
                           attention_heads=attention_heads, reference_loss_weight=reference_loss_weight,
                           latent_kl_weight=latent_kl_weight, chemical_regularization_weight=chemical_regularization_weight,
                           use_chemistry=use_chemistry, use_references=use_references, use_library=use_library)
        self.feature_dim = sum(map(len, feature_groups.values()))
        self.hidden_dim, self.latent_rank, self.residual_rank = hidden_dim, latent_rank, residual_rank
        self.min_scale, self.identity_transport = min_scale, identity_transport
        self.reference_loss_weight = reference_loss_weight
        self.chemical_regularization_weight = chemical_regularization_weight
        self.use_chemistry, self.use_references, self.use_library = use_chemistry, use_references, use_library
        if biology_kernel_mode not in {"off", "structured", "generic", "mlp"}:
            raise ValueError("Unknown chemical response kernel mode")
        if observation_family not in {"gaussian", "copula_t4"}:
            raise ValueError("Unknown joint observation family")
        self.observation_family = observation_family
        self.biology_kernel_mode = biology_kernel_mode
        if observation_family != "gaussian":
            self.config["observation_family"] = observation_family
        group_index = torch.empty(self.feature_dim, dtype=torch.long)
        for index, coordinates in enumerate(feature_groups.values()):
            group_index[list(coordinates)] = index
        self.register_buffer("feature_group_index", group_index)
        self.register_buffer("outcome_center",torch.zeros(self.feature_dim))
        self.register_buffer("outcome_scale",torch.ones(self.feature_dim))
        self.profile_encoder = GroupedProfileEncoder(feature_groups, hidden_dim,
                                                      group_attention_layers, attention_heads)
        self.reference_encoder = HierarchicalReferenceEncoder(reference_dim, hidden_dim)
        self.panel_encoder = IdentityMatchedReferenceEncoder(self.feature_dim, hidden_dim)
        self.condition_encoder = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.chemical_prior = ChemicalGaussianPrior(chemical_dim, hidden_dim, latent_rank)
        self.context_norm, self.query_norm = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.evidence_head = nn.Linear(hidden_dim, 2 * latent_rank)
        self.shared_context_encoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.Tanh())
        self.count_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        self.query, self.key = nn.Linear(hidden_dim, hidden_dim, bias=False), nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.operator = MeasurementKernelOperator(hidden_dim, kernel_mode, descriptor_kind="laws")
        self.prior_token = nn.Parameter(torch.randn(hidden_dim) * .02)
        self.library_attention = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
        self.library_null = nn.Parameter(torch.randn(1, 1, hidden_dim) * .02)
        self.library_density = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.library_global = nn.Sequential(nn.Linear(2 * self.feature_dim + 3, hidden_dim),
                                            nn.GELU(), nn.LayerNorm(hidden_dim))
        self.decoder = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.mean_head = nn.Linear(hidden_dim, self.feature_dim)
        self.transport_scale = nn.Linear(hidden_dim, self.feature_dim)
        # Unlike R1, initialization does not force p(future mean) to equal X.
        nn.init.normal_(self.transport_scale.weight, std=.02 / math.sqrt(hidden_dim))
        nn.init.zeros_(self.transport_scale.bias)
        self.scale_head = GroupHeteroscedasticScale(hidden_dim, group_index, min_scale)
        self.factor_heads = nn.ModuleList([StructuredFactorHead(hidden_dim, group_index, latent_rank) for _ in range(4)])
        self.residual_head = StructuredFactorHead(hidden_dim, group_index, residual_rank)
        self.biological_prior = None
        if not isinstance(use_biology_prior, bool):
            raise ValueError("use_biology_prior must be boolean")
        if use_biology_prior:
            from .biology import EvidenceAwareMechanismPrior
            # Optional prior initialization must not change the legacy random
            # stream, including all-unknown training episodes.
            with torch.random.fork_rng(devices=[]):
                self.biological_prior = EvidenceAwareMechanismPrior(biology_vocabulary, hidden_dim, latent_rank)
            self.config.update(use_biology_prior=True, biology_vocabulary=biology_vocabulary)
        elif biology_vocabulary is not None:
            raise ValueError("A biological vocabulary requires an enabled biological prior")
        self.chemistry_response_kernel = None
        if biology_kernel_mode != "off":
            if not use_chemistry or chemical_anchor_data is None:
                raise ValueError("A response kernel requires chemistry and training-fitted anchors")
            from .biology_kernel import ChemistryResponseKernelPrior
            # New-module initialization leaves all shared model tensors and the
            # subsequent training RNG stream identical to the no-kernel arm.
            with torch.random.fork_rng(devices=[]):
                self.chemistry_response_kernel = ChemistryResponseKernelPrior(
                    chemical_dim, hidden_dim, latent_rank, mode=biology_kernel_mode,
                    anchor_data=chemical_anchor_data)
            self.config.update(biology_kernel_mode=biology_kernel_mode,
                               chemical_anchor_data=chemical_anchor_data)
        elif chemical_anchor_data is not None:
            raise ValueError("Chemical anchors require an enabled response kernel")

    @staticmethod
    def gaussian_kl(qmean, qvar, pmean, pvar):
        return .5 * (torch.log(pvar / qvar) + (qvar + (qmean - pmean).square()) / pvar - 1).sum(-1)

    def freeze_encoder(self):
        self.profile_encoder.requires_grad_(False)
        self.profile_encoder.eval()

    def train(self,mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.profile_encoder.parameters()):
            self.profile_encoder.eval()
        return self

    def sample_joint(self,batch,n_samples,generator=None,environment_noise_cache=None):
        return self(batch).sample_joint(n_samples,generator,environment_noise_cache)

    def set_outcome_transform(self, center, scale):
        """Bind law descriptors to the declared outcome space, not training z-scores."""
        center=torch.as_tensor(center,dtype=self.outcome_center.dtype,device=self.outcome_center.device)
        scale=torch.as_tensor(scale,dtype=self.outcome_scale.dtype,device=self.outcome_scale.device)
        if center.shape != self.outcome_center.shape or scale.shape != self.outcome_scale.shape or torch.any(scale<=0):
            raise ValueError("Outcome affine transform must cover every coordinate with positive scale")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("Outcome transform must be finite")
        self.outcome_center.copy_(center); self.outcome_scale.copy_(scale)

    def _update_gaussian(self, prior_mean, prior_var, tokens, mask):
        location, raw_precision = self.evidence_head(tokens).chunk(2, -1)
        precision = (F.softplus(raw_precision) + 1e-5) * mask.unsqueeze(-1)
        posterior_precision = prior_var.reciprocal() + precision.sum(-2)
        variance = posterior_precision.reciprocal()
        mean = variance * (prior_mean / prior_var + (location * precision).sum(-2))
        # No observed evidence is exactly the supplied prior, not a sequence
        # of reciprocal operations that can drift by one floating-point ULP.
        # The same identity is used by every training-objective arm.
        available=mask.any(-1,keepdim=True)
        return torch.where(available,mean,prior_mean),torch.where(available,variance,prior_var)

    def _reference(self, batch, prefix, outer_mask=None):
        values, mask = batch[prefix + "_reference"], batch[prefix + "_reference_mask"]
        if outer_mask is not None:
            mask = mask & outer_mask.unsqueeze(-1)
        if not self.use_references:
            mask = torch.zeros_like(mask)
        encoded = self.reference_encoder(values, mask)
        shape = values.shape[:-2]
        one = values.new_ones((*shape, self.feature_dim))
        zero = values.new_zeros((*shape, self.feature_dim))
        loss, count = encoded.sum() * 0, torch.zeros((), dtype=torch.long, device=values.device)
        panel = batch.get(prefix + "_panel_y")
        compact = batch.get(prefix + "_panel_index")
        if (self.use_references and compact is not None and compact.shape[-1] > 0
                and len(batch.get("panel_catalog_y", ())) > 0):
            pmask = batch[prefix + "_panel_mask"]
            if outer_mask is not None:
                pmask = pmask & outer_mask[...,None,None]
            token,loss,one,zero,count=self.panel_encoder.forward_compact(
                batch["panel_catalog_y"],batch["panel_catalog_template"],batch["panel_catalog_template_mask"],compact,pmask,
                batch.get(prefix+"_panel_identity"))
            encoded=encoded+token
        elif self.use_references and panel is not None and panel.shape[-2] > 0:
            pmask = batch[prefix + "_panel_mask"]
            if outer_mask is not None:
                pmask = pmask & outer_mask[..., None, None]
            template_mask = batch.get(prefix + "_panel_template_mask", pmask)
            token, loss, one, zero, count = self.panel_encoder(
                panel, batch[prefix + "_panel_template"], pmask, template_mask,
                batch.get(prefix+"_panel_identity"))
            encoded = encoded + token
        return encoded, loss, one, zero, count

    def _library(self, batch, query, pooled):
        b, t, _ = query.shape
        y, mask = batch.get("library_y"), batch.get("library_mask")
        if not self.use_library or y is None or y.shape[1] == 0:
            return torch.zeros_like(query)
        if mask.dtype != torch.bool or mask.shape != y.shape[:2]:
            raise ValueError("library_mask must be boolean [B,L]")
        y = torch.where(mask.unsqueeze(-1), y, torch.zeros_like(y))
        cond = torch.where(mask.unsqueeze(-1), batch["library_cond"], torch.zeros_like(batch["library_cond"]))
        tokens = encode_library_profiles(self.profile_encoder,self.condition_encoder,y,cond,mask,
                                         batch.get("library_index"))
        null = self.library_null.expand(b, -1, -1)
        tokens = torch.cat((tokens, null), 1)
        padding = torch.cat((~mask, torch.zeros(b, 1, dtype=torch.bool, device=mask.device)), -1)
        attended, _ = self.library_attention(query, tokens, tokens, key_padding_mask=padding, need_weights=False)
        distance = (tokens[:, :-1] - pooled.unsqueeze(1)).square().mean(-1)
        count = mask.sum(-1).clamp_min(1)
        avg = (distance * mask).sum(-1) / count
        density = (torch.exp(-distance) * mask).sum(-1) / count
        features = torch.stack((avg, density, torch.log1p(mask.sum(-1).to(query.dtype))), -1)
        global_context = torch.zeros_like(pooled)
        if "library_global_mean" in batch:
            global_context = self.library_global(library_global_features(batch,mask.any(-1),query.dtype))
        return (attended + self.library_density(features).unsqueeze(1) + global_context.unsqueeze(1)) * mask.any(-1)[:, None, None]

    def _state_details(self, batch):
        if "target_y" in batch:
            raise ValueError("Target measurements must be a separate training argument")
        y, mask = batch["context_y"], batch["context_mask"]
        if y.ndim != 3 or mask.shape != y.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Invalid context shapes/mask")
        y = torch.where(mask.unsqueeze(-1), y, torch.zeros_like(y))
        cond = torch.where(mask.unsqueeze(-1), batch["context_cond"], torch.zeros_like(batch["context_cond"]))
        target_cond = batch["target_cond"]
        if not torch.isfinite(cond).all() or not torch.isfinite(target_cond).all():
            raise ValueError("Available measurement conditions must be finite")
        b, c, _ = y.shape
        t = target_cond.shape[1]
        chem_mask = batch.get("chem_mask", torch.ones(b, dtype=torch.bool, device=y.device))
        if not self.use_chemistry:
            chem_mask = torch.zeros_like(chem_mask)
        chemical, prior_mean, prior_var = self.chemical_prior(batch["chem"], chem_mask)
        if self.chemistry_response_kernel is not None:
            chemical, prior_mean, prior_var = self.chemistry_response_kernel(
                batch["chem"], chem_mask, chemical, prior_mean, prior_var)
        if self.biological_prior is not None:
            prior_mean, prior_var = self.biological_prior(batch, prior_mean, prior_var)
        cref, closs, cscale, coffset, cn = self._reference(batch, "context", mask)
        tref, tloss, tscale, toffset, tn = self._reference(batch, "target")
        context = self.context_norm(self.profile_encoder(y) + self.condition_encoder(cond) + cref + chemical.unsqueeze(1))
        pm, pv = self._update_gaussian(prior_mean, prior_var, context, mask)
        count = mask.sum(-1, keepdim=True)
        pooled = (context * mask.unsqueeze(-1)).sum(1) / count.clamp_min(1)
        pooled = torch.where(count > 0, pooled, self.prior_token + chemical)
        pooled = pooled + self.count_encoder(torch.log1p(count.to(y.dtype)))
        context_group, target_group = batch["context_group"], batch["target_group"]
        if context_group.shape != (b,c,3) or target_group.shape != (b,t,3) or target_group.dtype != torch.int64:
            raise ValueError("Hierarchical groups must be int64 [B,W,3]")
        same, sharing = mask[:, None, :].expand(-1,t,-1), []
        for level in range(3):
            left, right = target_group[...,level], context_group[...,level]
            same = same & (left.unsqueeze(-1) == right.unsqueeze(1)) & (left.unsqueeze(-1) >= 0)
            sharing.append(same)
        counts = torch.stack([s.sum(-1) for s in sharing], -1).to(y.dtype)
        query = self.query_norm(self.condition_encoder(target_cond) + tref + chemical.unsqueeze(1)
                                + self.shared_context_encoder(torch.log1p(counts)))
        library = self._library(batch, query, pooled)
        query = query + library
        prestate = self.decoder(torch.cat((query, pooled.unsqueeze(1).expand_as(query), library), -1))
        preloading = self.factor_heads[0](prestate) / math.sqrt(self.latent_rank)
        premean = self.mean_head(prestate) + torch.einsum("btdr,br->btd", preloading, pm)
        oscale=self.outcome_scale
        noise = (self.scale_head(prestate)*oscale).square().mean(-1)
        environmental = [head(prestate)*oscale[:,None] / math.sqrt(self.latent_rank) for head in self.factor_heads[1:]]
        envvar = [load.square().sum(-1).mean(-1) for load in environmental]
        noise = noise + sum(envvar) + (self.residual_head(prestate)*oscale[:,None]).square().mean((-1,-2))
        rho = (premean*oscale+self.outcome_center).square().mean(-1) / noise.clamp_min(1e-8)
        if c:
            scores = torch.einsum("bth,bch->btc", self.query(query), self.key(context)) / math.sqrt(self.hidden_dim)
            scores = scores.masked_fill(~mask.unsqueeze(1), -torch.finfo(scores.dtype).max)
            attention = scores.softmax(-1) * mask.unsqueeze(1)
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
            cd = (target_cond.unsqueeze(2) - cond.unsqueeze(1)).square().mean(-1).sqrt()
            coavailable = batch["target_reference_mask"].unsqueeze(2) & (batch["context_reference_mask"] & mask.unsqueeze(-1)).unsqueeze(1)
            if not self.use_references:
                coavailable = torch.zeros_like(coavailable)
            craw = torch.where((batch["context_reference_mask"] & mask.unsqueeze(-1)).unsqueeze(-1), batch["context_reference"], torch.zeros_like(batch["context_reference"]))
            traw = torch.where(batch["target_reference_mask"].unsqueeze(-1), batch["target_reference"], torch.zeros_like(batch["target_reference"]))
            rd = (((traw.unsqueeze(2)-craw.unsqueeze(1)).square().mean(-1) * coavailable).sum(-1)
                  / coavailable.sum(-1).clamp_min(1)).sqrt()
            context_environmental=[head(context)*oscale[:,None]/math.sqrt(self.latent_rank) for head in self.factor_heads[1:]]
            context_noise=(self.scale_head(context)*oscale).square().mean(-1)
            context_noise=context_noise+sum(load.square().sum(-1).mean(-1) for load in context_environmental)
            context_noise=context_noise+(self.residual_head(context)*oscale[:,None]).square().mean((-1,-2))
            covariance=sum(torch.einsum("btdr,bcdr->btc",left,right)/self.feature_dim * s
                           for left,right,s in zip(environmental,context_environmental,sharing))
            correlation=covariance/torch.sqrt(noise.unsqueeze(-1)*context_noise.unsqueeze(1)).clamp_min(1e-8)
            n_cells = batch.get("context_n_cells", y.new_zeros((b,c)))
            n_mask = batch.get("context_n_cells_mask", torch.zeros_like(mask)) & mask
            n_cells = torch.where(n_mask, n_cells, torch.ones_like(n_cells))
            descriptors = torch.stack((rho.unsqueeze(-1).expand(-1,-1,c), correlation,
                                      count[:,None,:].expand(-1,t,c).to(y.dtype),
                                      n_cells[:,None,:].expand(-1,t,-1), n_mask[:,None,:].expand(-1,t,-1).to(y.dtype),
                                      cd, rd, coavailable.any(-1).to(y.dtype),
                                      pv.mean(-1)[:,None,None].expand(-1,t,c)), -1)
            messages = self.operator(context.unsqueeze(1).expand(-1,t,-1,-1),
                                     query.unsqueeze(2).expand(-1,-1,c,-1), descriptors)
            attended = (attention.unsqueeze(-1) * messages).sum(-2)
            # Identity-matched reference affine transport is deterministic;
            # an unconstrained learned residual path may correct it.
            clean_y=torch.where(torch.isfinite(y),y,torch.zeros_like(y))
            canonical = (clean_y - coffset) / cscale
            available = torch.isfinite(y) & mask.unsqueeze(-1)
            weights = attention.unsqueeze(-1) * available.unsqueeze(1)
            weights = weights / weights.sum(-2,keepdim=True).clamp_min(1e-12)
            raw_transport = (weights * canonical.unsqueeze(1)).sum(-2) * tscale + toffset
        else:
            attended = torch.zeros_like(query)
            raw_transport = y.new_zeros((b,t,self.feature_dim))
            descriptors = y.new_zeros((b,t,0,9))
        state = self.decoder(torch.cat((query, attended, pooled.unsqueeze(1).expand(-1,t,-1) + library), -1))
        return dict(state=state, query=query, raw_transport=raw_transport, context=context,
                    prior_mean=prior_mean, prior_var=prior_var, posterior_mean=pm, posterior_var=pv,
                    reference_loss=(closs*cn+tloss*tn)/(cn+tn).clamp_min(1), reference_count=cn+tn,
                    descriptors=descriptors, rho=rho, model_noise=noise, chemical=chemical, target_reference=tref)

    def _state(self, batch):
        details = self._state_details(batch)
        return details["state"], details["query"], details["raw_transport"]

    def descriptors(self, batch):
        """Model-implied law inputs; not an identifiability certificate."""
        return self._state_details(batch)["descriptors"]

    def _decode(self, batch, details, latent_mean, latent_var=None):
        state, query = details["state"], details["query"]
        b,t,_ = state.shape
        group = batch["target_group"]
        mean = self.mean_head(state)
        if self.identity_transport:
            mean = mean + self.transport_scale(query) * details["raw_transport"]
        load = self.factor_heads[0](state) / math.sqrt(self.latent_rank)
        mean = mean + torch.einsum("btdr,br->btd",load,latent_mean)
        compound = torch.zeros_like(load) if latent_var is None else load * latent_var.sqrt()[:,None,None,:]
        parts, environmental = [compound], []
        for level in range(3):
            factor = self.factor_heads[level+1](state) / math.sqrt(self.latent_rank)
            environmental.append(factor)
            same = torch.ones((b,t,t),dtype=torch.bool,device=state.device)
            for ancestor in range(level+1):
                ids=group[...,ancestor]
                same = same & (ids.unsqueeze(-1)==ids.unsqueeze(-2)) & (ids.unsqueeze(-1)>=0)
            same = same | torch.eye(t,dtype=torch.bool,device=state.device).unsqueeze(0)
            assignment = F.one_hot(same.to(torch.int64).argmax(-1),t).to(state.dtype)
            parts.append((factor.unsqueeze(-2)*assignment[:,:,None,:,None]).reshape(b,t,self.feature_dim,t*self.latent_rank))
        residual = self.residual_head(state) / math.sqrt(self.residual_rank)
        assignment = torch.eye(t,device=state.device,dtype=state.dtype)
        residual = (residual.unsqueeze(-2)*assignment[None,:,None,:,None]).reshape(b,t,self.feature_dim,t*self.residual_rank)
        parts.append(residual)
        return JointGaussian(mean,self.scale_head(state).square(),torch.cat(parts,-1),
                             torch.cat((compound,residual),-1),tuple(environmental),group,
                             "conditional_NP_chemical_latent_and_hierarchical_measurement_factors")

    def forward(self,batch):
        details=self._state_details(batch)
        return self._observation_distribution(self._decode(
            batch,details,details["posterior_mean"],details["posterior_var"]))

    def _observation_distribution(self, gaussian):
        if self.observation_family == "gaussian":
            return gaussian
        from .copula import StudentT4GaussianCopula
        return StudentT4GaussianCopula(gaussian)

    def loss(self,batch,target_y,target_mask=None,*,objective="elbo"):
        if objective not in {"elbo", "predictive_nll"}:
            raise ValueError("Unknown probability training objective")
        if self.observation_family != "gaussian" and objective != "predictive_nll":
            raise ValueError("Copula observations require the complete predictive_nll objective")
        details=self._state_details(batch)
        predictive=self._observation_distribution(self._decode(
            batch,details,details["posterior_mean"],details["posterior_var"]))
        observed=predictive.observed_mask(target_y,target_mask)
        counts=observed.sum()
        if counts == 0:
            raise ValueError("Training requires measured targets")
        if objective == "predictive_nll":
            # Latents are integrated in the Gaussian dependence model. A
            # declared copula family adds its invertible marginal map/Jacobian.
            # Do not construct target-encoded q(z|S,T) for this objective: the
            # predictive state is exclusively a function of legal inputs S.
            joint_log_prob=predictive.joint_log_prob(target_y,target_mask)
            nll=-joint_log_prob/counts
            chemical_kl=self.gaussian_kl(details["posterior_mean"],details["posterior_var"],
                                          details["prior_mean"],details["prior_var"]).mean()
            total=(nll+self.reference_loss_weight*details["reference_loss"]
                   +self.chemical_regularization_weight*chemical_kl)
            return {"loss":total,"nll":nll,"chemical_kl_regularizer":chemical_kl,
                    "reference_reconstruction":details["reference_loss"],
                    "reference_reconstruction_count":details["reference_count"],
                    "joint_log_prob":joint_log_prob,
                    "mean_marginal_variance":predictive.marginal_variance.mean()}
        clean=torch.where(observed,target_y,torch.full_like(target_y,float("nan")))
        target_token=self.context_norm(self.profile_encoder(clean)+self.condition_encoder(batch["target_cond"])
                                       +details["target_reference"]+details["chemical"].unsqueeze(1))
        qm,qv=self._update_gaussian(details["posterior_mean"],details["posterior_var"],target_token,observed.any(-1))
        kl=self.gaussian_kl(qm,qv,details["posterior_mean"],details["posterior_var"]).sum()
        # One reparameterized draw is an unbiased estimator of the conditional
        # reconstruction expectation. It is NOT log E_q p(Y|z,S).
        z=qm+qv.sqrt()*torch.randn_like(qm)
        conditional=self._decode(batch,details,z,None)
        reconstruction=-conditional.joint_log_prob(target_y,target_mask)
        elbo=(reconstruction+kl)/counts
        chemical_kl=self.gaussian_kl(details["posterior_mean"],details["posterior_var"],
                                      details["prior_mean"],details["prior_var"]).mean()
        joint_log_prob=predictive.joint_log_prob(target_y,target_mask)
        nll=-joint_log_prob/counts
        total=elbo+self.reference_loss_weight*details["reference_loss"]+self.chemical_regularization_weight*chemical_kl
        return {"loss":total,"nll":nll,"elbo_loss":elbo,"latent_kl":kl/counts,
                "chemical_kl_regularizer":chemical_kl,"reference_reconstruction":details["reference_loss"],
                "reference_reconstruction_count":details["reference_count"],"joint_log_prob":joint_log_prob,
                "mean_marginal_variance":predictive.marginal_variance.mean()}

    @torch.no_grad()
    def variance_components(self,batch):
        """Gaussian components; copula keys describe latent dependence, not Y effects."""
        details = self._state_details(batch)
        distribution = self._decode(batch, details, details["posterior_mean"], details["posterior_var"])
        rank=self.latent_rank
        local=distribution.local_factors
        components={"compound_latent":local[...,:rank].square().sum(-1),
                    "within_well_low_rank":local[...,rank:].square().sum(-1),
                    "diagonal":distribution.diag_var}
        for name,factor in zip(("source","batch","plate"),distribution.environment_loadings):
            components[name]=factor.square().sum(-1)
        if self.observation_family != "gaussian":
            return {"gaussian_copula_latent__" + key: value for key, value in components.items()}
        return components
