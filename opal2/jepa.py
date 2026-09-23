"""Conditional cross-repeat JEPA with an actual EMA target network.

This module trains representations, not a calibrated predictive distribution.
The second stage fits the measurement likelihood in fixed measured coordinates.
"""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from .model import (GroupedProfileEncoder, HierarchicalReferenceEncoder,
                    IdentityMatchedReferenceEncoder, encode_library_profiles, library_global_features)


class ConditionalJEPA(nn.Module):
    """Predict withheld-repeat teacher embeddings from only observed context.

    An EMA teacher receives target profiles under stop-gradient. Student-view
    variance and covariance penalties discourage collapse. Availability masks
    are preserved; future measured target values enter only the training loss.
    Call update_teacher after each optimizer step, never before/backprop through
    it. frozen_encoder exports the EMA teacher for stage-two likelihood fitting.
    """
    def __init__(self, encoder: GroupedProfileEncoder, condition_dim: int,
                 reference_dim: int, chemical_dim: int, hidden_dim: int | None = None,
                 ema_decay: float = 0.99, alignment_weight: float = 25.0,
                 variance_weight: float = 25.0, covariance_weight: float = 1.0,
                 use_chemistry: bool = True, use_references: bool = True, use_library: bool = True):
        super().__init__()
        hidden_dim = encoder.hidden_dim if hidden_dim is None else hidden_dim
        if hidden_dim != encoder.hidden_dim or not 0 <= ema_decay < 1:
            raise ValueError("JEPA encoder size or EMA decay is invalid")
        self.hidden_dim, self.ema_decay = hidden_dim, ema_decay
        self.use_chemistry,self.use_references,self.use_library=use_chemistry,use_references,use_library
        self.alignment_weight = alignment_weight
        self.variance_weight, self.covariance_weight = variance_weight, covariance_weight
        self.student_encoder = encoder
        self.student_projector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                                nn.Linear(hidden_dim, hidden_dim))
        self.teacher_encoder = copy.deepcopy(encoder).requires_grad_(False)
        self.teacher_projector = copy.deepcopy(self.student_projector).requires_grad_(False)
        self.reference_encoder = HierarchicalReferenceEncoder(reference_dim, hidden_dim)
        self.panel_encoder = IdentityMatchedReferenceEncoder(encoder.feature_dim,hidden_dim)
        self.condition_encoder = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.GELU())
        self.chemical_encoder = nn.Sequential(nn.Linear(chemical_dim, hidden_dim), nn.GELU())
        self.missing_chemistry = nn.Parameter(torch.zeros(hidden_dim))
        self.library_attention = nn.MultiheadAttention(hidden_dim,encoder.attention_heads,batch_first=True)
        self.library_null = nn.Parameter(torch.zeros(1,1,hidden_dim))
        self.library_global = nn.Sequential(nn.Linear(2*encoder.feature_dim+3,hidden_dim),nn.GELU())
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.predictor = nn.Sequential(nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.GELU(),
                                       nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim))
        self.prior_token = nn.Parameter(torch.zeros(hidden_dim))
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.int64))
        self.teacher_encoder.eval()
        self.teacher_projector.eval()

    def train(self, mode: bool = True) -> "ConditionalJEPA":
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_projector.eval()
        return self

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Only conditional predicted embeddings, no target observations."""
        if "target_y" in batch:
            raise ValueError("Target measurements must be a separate training argument")
        mask = batch["context_mask"]
        y = torch.where(mask.unsqueeze(-1), batch["context_y"],
                        torch.zeros_like(batch["context_y"]))
        cond = torch.where(mask.unsqueeze(-1), batch["context_cond"],
                           torch.zeros_like(batch["context_cond"]))
        context = self.student_projector(self.student_encoder(y))
        chem_mask=batch.get("chem_mask",torch.ones(y.shape[0],dtype=torch.bool,device=y.device))
        if not self.use_chemistry: chem_mask=torch.zeros_like(chem_mask)
        if chem_mask.dtype != torch.bool or chem_mask.shape != y.shape[:1]:
            raise ValueError("chem_mask must be boolean [B]")
        chem=torch.where(chem_mask.unsqueeze(-1),batch["chem"],torch.zeros_like(batch["chem"]))
        if not torch.isfinite(chem).all():
            raise ValueError("Available chemical descriptors must be finite")
        chemical = torch.where(chem_mask.unsqueeze(-1),self.chemical_encoder(chem),self.missing_chemistry)
        cm=batch["context_reference_mask"] & mask.unsqueeze(-1)
        tm=batch["target_reference_mask"]
        if not self.use_references: cm,tm=torch.zeros_like(cm),torch.zeros_like(tm)
        cref = self.reference_encoder(batch["context_reference"],cm)
        tref = self.reference_encoder(batch["target_reference"],tm)
        cref=cref+self._panels(batch,"context",mask)
        tref=tref+self._panels(batch,"target")
        context = context + self.condition_encoder(cond) + cref
        query = self.condition_encoder(batch["target_cond"]) + tref + chemical.unsqueeze(1)
        if context.shape[1]:
            scores = torch.einsum("bth,bch->btc", self.query(query), self.key(context)) / math.sqrt(self.hidden_dim)
            scores = scores.masked_fill(~mask.unsqueeze(1), -torch.finfo(scores.dtype).max)
            weights = scores.softmax(-1) * mask.unsqueeze(1)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            attended = weights @ context
        else:
            attended=torch.zeros_like(query)
        count = mask.sum(-1, keepdim=True)
        pooled = (context * mask.unsqueeze(-1)).sum(1) / count.clamp_min(1)
        pooled = torch.where(count > 0, pooled, self.prior_token + chemical)
        ly,lm=batch.get("library_y"),batch.get("library_mask")
        if self.use_library and ly is not None and ly.shape[1]:
            ly=torch.where(lm.unsqueeze(-1),ly,torch.zeros_like(ly))
            lc=torch.where(lm.unsqueeze(-1),batch["library_cond"],torch.zeros_like(batch["library_cond"]))
            tokens=encode_library_profiles(self.student_encoder,self.condition_encoder,ly,lc,lm,
                                           batch.get("library_index"))
            tokens=torch.cat((tokens,self.library_null.expand(len(ly),-1,-1)),1)
            padding=torch.cat((~lm,torch.zeros(len(ly),1,dtype=torch.bool,device=ly.device)),1)
            library,_=self.library_attention(query,tokens,tokens,key_padding_mask=padding,need_weights=False)
            if "library_global_mean" in batch:
                global_features=library_global_features(batch,lm.any(-1),y.dtype)
                library=library+self.library_global(global_features).unsqueeze(1)
            query=query+library*lm.any(-1)[:,None,None]
        return self.predictor(torch.cat((query, attended,
                                         pooled.unsqueeze(1).expand_as(query)), -1))

    def _panels(self,batch,prefix,outer_mask=None):
        shape=batch[prefix+"_cond"].shape[:-1]
        output=batch[prefix+"_cond"].new_zeros((*shape,self.hidden_dim))
        if not self.use_references: return output
        compact=batch.get(prefix+"_panel_index")
        panel=batch.get(prefix+"_panel_y")
        if compact is not None and compact.shape[-1] and len(batch.get("panel_catalog_y",())):
            mask=batch[prefix+"_panel_mask"]
            if outer_mask is not None: mask=mask&outer_mask[...,None,None]
            output,*_=self.panel_encoder.forward_compact(batch["panel_catalog_y"],batch["panel_catalog_template"],
                        batch["panel_catalog_template_mask"],compact,mask,batch.get(prefix+"_panel_identity"))
        elif panel is not None and panel.shape[-2]:
            mask=batch[prefix+"_panel_mask"]
            if outer_mask is not None: mask=mask&outer_mask[...,None,None]
            output,*_=self.panel_encoder(panel,batch[prefix+"_panel_template"],mask,
                          batch.get(prefix+"_panel_template_mask",mask),batch.get(prefix+"_panel_identity"))
        return output

    def loss(self, batch: dict[str, torch.Tensor], target_y: torch.Tensor,
             target_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if target_mask is not None:
            if target_mask.dtype != torch.bool:
                raise ValueError("JEPA target mask must be boolean")
            if target_mask.shape == target_y.shape:
                target_y = torch.where(target_mask, target_y, torch.full_like(target_y, float("nan")))
                target_mask = target_mask.any(-1)
            if target_mask.shape != target_y.shape[:2]:
                raise ValueError("JEPA target mask must index target wells or coordinates")
        # Validity must be recomputed AFTER coordinate masking. A well whose
        # only finite coordinate was hidden is not a measured teacher view.
        valid = torch.isfinite(target_y).any(-1)
        if target_mask is not None:
            valid = valid & target_mask
        if not valid.any():
            raise ValueError("JEPA needs observed target views")
        predicted = self(batch)
        with torch.no_grad():
            teacher = self.teacher_projector(self.teacher_encoder(target_y))
        alignment = F.mse_loss(predicted[valid], teacher[valid])
        # Student target views are for self-supervised training only. They never
        # enter the context predictor or the deployed measurement-model input.
        context_views = self.student_projector(self.student_encoder(batch["context_y"]))
        target_views = self.student_projector(self.student_encoder(target_y))
        embeddings = torch.cat((context_views[batch["context_mask"]], target_views[valid]), 0)
        if embeddings.shape[0] < 2:
            raise ValueError("Variance/covariance anti-collapse needs at least two valid views")
        centered = embeddings - embeddings.mean(0, keepdim=True)
        covariance = centered.T @ centered / (embeddings.shape[0] - 1)
        std = torch.sqrt(covariance.diagonal() + 1e-4)
        variance = F.relu(1 - std).mean()
        off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
        covariance_penalty = off_diagonal.square().sum() / embeddings.shape[1]
        total = (self.alignment_weight * alignment + self.variance_weight * variance
                 + self.covariance_weight * covariance_penalty)
        return {"loss": total, "alignment": alignment, "variance": variance,
                "covariance": covariance_penalty, "embedding_std": std.mean()}

    @torch.no_grad()
    def update_teacher(self) -> None:
        for student, teacher in ((self.student_encoder, self.teacher_encoder),
                                 (self.student_projector, self.teacher_projector)):
            for source, target in zip(student.parameters(), teacher.parameters(), strict=True):
                target.mul_(self.ema_decay).add_(source, alpha=1 - self.ema_decay)
            for source, target in zip(student.buffers(), teacher.buffers(), strict=True):
                if target.is_floating_point():
                    target.mul_(self.ema_decay).add_(source, alpha=1 - self.ema_decay)
                else:
                    target.copy_(source)
        self.ema_updates.add_(1)

    def frozen_encoder(self) -> GroupedProfileEncoder:
        return copy.deepcopy(self.teacher_encoder).eval().requires_grad_(False)
