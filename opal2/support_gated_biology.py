"""Opt-in relation masking and deterministic support shrinkage for EU/Rx R3.

Historical adapters and artifacts keep their original semantics. The new
adapter preserves the 24-input / 146-parameter right branch and its optimizer.
MASK_ONLY removes every contribution from an unavailable relation, including
bases nonzero at raw zero. MASK_CONFIDENCE additionally multiplies the bounded
right output by the largest confidence among available relations. This is an
overall support gate, not an independently learned per-relation gate.

The existing two-output bias is retained, but is zero when no relation is
available and shrunk by the same confidence gate. It is explicitly a supported
population intercept; it is not assigned to a missing target or MoA channel.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .dual_branch_biology import (
    BOUND, BiologicalBasisBranch, _fit, _matrix, _scaling, _support,
)
from .eu_r3_adapters import EUResidualAdapter, SUPPORT_SEMANTICS


GATE_MODES = ("MASK_ONLY", "MASK_CONFIDENCE")
ADAPTER_VERSION = "relation_masked_support_gate_v1"


class MaskedBiologicalBasisBranch(BiologicalBasisBranch):
    """Same trainable parameters; missing-relation fields cannot contribute."""

    def __init__(self, names, mode, *, gate_mode="MASK_ONLY"):
        if gate_mode not in GATE_MODES:
            raise ValueError("Unknown deterministic biology gate mode")
        super().__init__(names, mode)
        self.gate_mode = gate_mode
        self.relations = tuple(dict.fromkeys(name.split("_", 1)[0] for name in names))
        if not set(self.relations) <= {"target", "moa"}:
            raise ValueError("Every biological field must have target_ or moa_ prefix")
        if len(set(names)) != len(names):
            raise ValueError("Biological field names must be unique")
        self.available_indices = []
        self.confidence_indices = []
        self.field_relation_indices = []
        for relation in self.relations:
            try:
                self.available_indices.append(names.index(relation + "_available"))
                self.confidence_indices.append(names.index(relation + "_confidence"))
            except ValueError as exc:
                raise ValueError("Each relation requires available and confidence fields") from exc
        for name in names:
            self.field_relation_indices.append(self.relations.index(name.split("_", 1)[0]))

    def relation_support(self, raw):
        available = raw[:, self.available_indices]
        if not torch.all((available == 0) | (available == 1)):
            raise ValueError("Relation availability must be exactly zero or one")
        return available.bool()

    def output_gate(self, raw):
        """Max is an OR-style support rule; missing relations cannot dilute it."""
        available = self.relation_support(raw)
        if self.gate_mode == "MASK_ONLY":
            return available.any(dim=1).to(raw.dtype)
        confidence = raw[:, self.confidence_indices]
        active_confidence = torch.where(available, confidence, torch.zeros_like(confidence))
        if not torch.isfinite(active_confidence).all() or not torch.all(
            (active_confidence >= 0) & (active_confidence <= 1)
        ):
            raise ValueError("Active relation confidence must be finite and in [0, 1]")
        # The feature builder supplies ESS/(ESS+5) * (1-exp(-similarity_mass)).
        # Reusing that exact feature introduces no learned gate parameters.
        return active_confidence.max(dim=1).values

    def forward(self, raw, standardized):
        if raw.shape != standardized.shape or raw.ndim != 2 or raw.shape[1] != len(self.names):
            raise ValueError("Biological descriptor shape mismatch")
        available = self.relation_support(raw)
        fields_available = available[:, self.field_relation_indices]
        basis = self.basis(raw, standardized)
        basis = torch.where(fields_available[..., None], basis, torch.zeros_like(basis))
        local = (basis * self.local_coefficients).sum(-1)
        values = BOUND * torch.tanh(self.readout(local) / BOUND)
        gate = self.output_gate(raw)
        return torch.where(gate[:, None] > 0, values * gate[:, None], torch.zeros_like(values))


class SupportGatedBiologyAdapter(EUResidualAdapter):
    """Explicit new artifact type; never silently reinterpret a legacy file."""

    def __init__(self, empirical_dim, biological_names=(), *, mode="none", hidden_dim=16,
                 gate_mode="MASK_ONLY"):
        if gate_mode not in GATE_MODES:
            raise ValueError("Unknown deterministic biology gate mode")
        super().__init__(empirical_dim, biological_names, mode=mode, hidden_dim=hidden_dim)
        if self.right is not None:
            # Preserve identical initialization for a given seed despite the
            # new branch type: copy the already initialized legacy parameters.
            initialized = deepcopy(self.right.state_dict())
            self.right = MaskedBiologicalBasisBranch(list(biological_names), mode,
                                                     gate_mode=gate_mode).double()
            self.right.load_state_dict(initialized)
        self.config["gate_mode"] = gate_mode
        self.report = dict(support_semantics=SUPPORT_SEMANTICS,
                           adapter_version=ADAPTER_VERSION, gate_mode=gate_mode)

    @classmethod
    def load(cls, path):
        saved = torch.load(Path(path), map_location="cpu", weights_only=True)
        if (saved.get("report", {}).get("adapter_version") != ADAPTER_VERSION
                or saved.get("config", {}).get("gate_mode") not in GATE_MODES):
            raise ValueError("Artifact is not an explicit support-gated adapter")
        model = super().load(path)
        return model


def from_legacy(adapter, *, gate_mode="MASK_ONLY"):
    """Explicit same-weight diagnostic conversion, not retraining or adoption."""
    if not isinstance(adapter, EUResidualAdapter):
        raise TypeError("Supply an EU/Rx independent-support adapter")
    config = dict(adapter.config)
    config["gate_mode"] = gate_mode
    with torch.random.fork_rng():
        model = SupportGatedBiologyAdapter(**config)
    model.load_state_dict(adapter.state_dict())
    model.report = dict(deepcopy(adapter.report), support_semantics=SUPPORT_SEMANTICS,
                        adapter_version=ADAPTER_VERSION, gate_mode=gate_mode,
                        same_weight_diagnostic_conversion=True)
    return model.eval().requires_grad_(False)


def fit_right(left, empirical, biological, names, support, energies, ids,
              mode="structured", seed=20260917, *, gate_mode="MASK_ONLY"):
    """Fit only the new right branch, unchanged 60-epoch recipe and fixed left."""
    if not isinstance(left, EUResidualAdapter) or left.config["mode"] != "none":
        raise TypeError("Supply the common fitted EU R3 left-only adapter")
    x, b = _matrix(empirical, "empirical"), _matrix(biological, "biological")
    e = _matrix(energies, "energies")
    mask = _support(support, len(x))
    names, identifiers = list(names), np.asarray(ids, str)
    if (identifiers.shape != (len(x),) or len(set(identifiers)) != len(identifiers)
            or identifiers.tolist() != left.report.get("fitting_ids")):
        raise ValueError("Right fitting IDs must equal the full left REF identity sequence")
    if x.shape[1] != left.config["empirical_dim"]:
        raise ValueError("Empirical descriptor shape mismatch")
    if b.shape != (len(x), len(names)) or len(set(names)) != len(names):
        raise ValueError("Biological feature names must uniquely align")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = SupportGatedBiologyAdapter(x.shape[1], names, mode=mode,
            hidden_dim=left.config["hidden_dim"], gate_mode=gate_mode)
    if model.right is None:
        raise ValueError("The right branch requires generic or structured mode")
    available = model.right.relation_support(torch.as_tensor(b, dtype=torch.float64)).any(1).numpy()
    mask = mask & available
    model.left.load_state_dict(left.left.state_dict())
    model.empirical_center.copy_(left.empirical_center)
    model.empirical_scale.copy_(left.empirical_scale)
    center, scale = _scaling(b[mask])
    model.biological_center.copy_(torch.as_tensor(center))
    model.biological_scale.copy_(torch.as_tensor(scale))
    model.report = dict(model.report, left_report=deepcopy(left.report),
        biological_feature_names=names, biological_feature_kinds=list(model.right.kinds),
        right_trainable_parameters=sum(p.numel() for p in model.right.parameters()),
        empirical_fitted_on_all_supplied_rows=True,
        base_scale="ones; energies already standardized against CORE",
        relation_mask="all local basis contributions masked by relation_available",
        bias_policy="shared supported-population intercept; same availability/confidence gate as output",
        confidence_policy=("max existing confidence among available relations after bounded output"
                           if gate_mode == "MASK_CONFIDENCE" else "availability only"),
        confidence_formula="ESS/(ESS+5) * (1-exp(-similarity_mass))",
        missing_relation_has_no_parameters_added=True,
        basis_interpretation="Support/error response bases, not established pharmacological laws")
    return _fit(model, x, b, mask, e, np.ones_like(e), identifiers,
                branch="right", seed=seed, max_epochs=60)
