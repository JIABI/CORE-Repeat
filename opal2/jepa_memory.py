"""Memory-bounded execution of the unchanged grouped profile encoder.

Only the independent profile axis is chunked. Feature groups, attention layers,
parameters and profile outputs are unchanged. JEPA still receives all profiles
and computes its single full-minibatch variance/covariance objective afterwards.
"""
from __future__ import annotations

from numbers import Integral
from typing import Mapping, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from .model import GroupedProfileEncoder


class MemoryEfficientGroupedProfileEncoder(GroupedProfileEncoder):
    """Recompute per-profile activations instead of retaining the full batch.

    There is no profile-to-profile attention or BatchNorm in the base encoder;
    all attention operates among the feature groups of one profile. Therefore
    chunking this axis preserves its function. Non-reentrant checkpointing
    preserves parameter gradients even when measured inputs require no gradient.

    No parameters/buffers/modules are added, so state-dictionary keys are exactly
    compatible with GroupedProfileEncoder. Deep-copied EMA teachers retain the
    execution policy and use ordinary chunked evaluation under no_grad.
    """

    def __init__(self, feature_groups: Mapping[str, Sequence[int]], hidden_dim: int = 256,
                 attention_layers: int = 2, attention_heads: int = 4, *,
                 profile_chunk_size: int = 16):
        if (isinstance(profile_chunk_size, bool) or not isinstance(profile_chunk_size, Integral)
                or profile_chunk_size < 1):
            raise ValueError("profile_chunk_size must be a positive integer")
        super().__init__(feature_groups, hidden_dim, attention_layers, attention_heads)
        self.profile_chunk_size = int(profile_chunk_size)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim < 1 or y.shape[-1] != self.feature_dim:
            raise ValueError("Profile coordinate dimension mismatch")
        leading = y.shape[:-1]
        profiles = y.reshape(-1, self.feature_dim)
        if len(profiles) == 0:
            return super().forward(y)
        encode = super().forward
        use_checkpoint = self.training and torch.is_grad_enabled()
        outputs = []
        for block in profiles.split(self.profile_chunk_size, dim=0):
            if use_checkpoint:
                outputs.append(checkpoint(encode, block, use_reentrant=False,
                                          preserve_rng_state=True))
            else:
                outputs.append(encode(block))
        return torch.cat(outputs, dim=0).reshape(*leading, self.hidden_dim)
