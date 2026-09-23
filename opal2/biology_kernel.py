"""Training-bound chemical-similarity response bases for a latent prior.

This branch uses chemical structures, not unverified targets or pathways. Its
Tanimoto locality bases are a declared structural inductive bias, not binding
or pharmacodynamic laws. Observed cell profiles do not enter this chemical
prior; their information belongs in the world's context/posterior path.
"""
from __future__ import annotations

import copy
import math
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .kernels import SplineKANLinear


ANCHOR_SCHEMA_VERSION = 1


def _feature_geometry(chemical_dim, chemical_metadata):
    metadata = dict(chemical_metadata or {})
    if "fingerprint_indices" in metadata:
        raw_indices = metadata["fingerprint_indices"]
        if (not isinstance(raw_indices, (list, tuple)) or not raw_indices
                or any(isinstance(i, bool) or not isinstance(i, Integral) for i in raw_indices)):
            raise ValueError("fingerprint_indices must explicitly name integer chemical coordinates")
        indices = [int(i) for i in raw_indices]
        validity = metadata.get("validity_index")
    elif "bits" in metadata:
        bits = metadata["bits"]
        if isinstance(bits, bool) or not isinstance(bits, Integral) or bits < 1:
            raise ValueError("Declared fingerprint bits must be positive")
        indices = list(range(int(bits)))
        validity = (int(bits) if metadata.get("final_coordinate") == "valid_SMILES_indicator"
                    else metadata.get("validity_index"))
    elif chemical_dim in (512, 513):
        indices = list(range(512))
        validity = 512 if chemical_dim == 513 else None
    else:
        raise ValueError("Nonstandard chemical geometry requires explicit fingerprint metadata")
    if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= chemical_dim:
        raise ValueError("Fingerprint coordinates must be unique and in range")
    if validity is not None:
        if isinstance(validity, bool) or not isinstance(validity, Integral) or not 0 <= validity < chemical_dim:
            raise ValueError("Validity coordinate must be an integer in range")
        validity = int(validity)
        if validity in indices:
            raise ValueError("Validity indicator must not be a fingerprint bit")
    description = {"kind": str(metadata.get("kind", "declared binary fingerprint")),
                   "radius": metadata.get("radius"), "bits": len(indices),
                   "rdkit_version": metadata.get("rdkit_version")}
    for key in ("radius",):
        if description[key] is not None:
            if isinstance(description[key], bool) or not isinstance(description[key], Integral) or description[key] < 0:
                raise ValueError("Fingerprint radius must be a declared nonnegative integer")
            description[key] = int(description[key])
    if description["rdkit_version"] is not None:
        description["rdkit_version"] = str(description["rdkit_version"])
    return indices, validity, description


def _binary_fingerprints(chemical, mask, fingerprint_indices, validity_index):
    """Inspect selected rows only; missing fingerprint payloads are ignored."""
    active = mask.copy()
    if validity_index is not None:
        flag = chemical[:, validity_index]
        if np.any(active & (~np.isfinite(flag) | ((flag != 0) & (flag != 1)))):
            raise ValueError("Available structure validity indicators must be binary")
        active &= flag == 1
    values = chemical[:, fingerprint_indices]
    if np.any(~np.isfinite(values[active])) or np.any((values[active] != 0) & (values[active] != 1)):
        raise ValueError("Available fingerprints must contain binary finite bits")
    clean = np.where(active[:, None], values, 0.).astype(np.uint8)
    active &= clean.sum(axis=1) > 0
    return np.where(active[:, None], clean, 0).astype(np.uint8), active


def fit_chemical_anchors(chem, mask, ids, train_ids, chemical_metadata=None, max_anchors=64):
    """Select deterministic farthest-first binary-Tanimoto training landmarks.

No outcomes, assay labels, held-out structures or validation scores are used.
Duplicate fingerprints share one landmark represented by the lexicographically
first training identity. All used training fingerprints/masks are retained in
the JSON-compatible result, allowing resume to compare a fresh fit explicitly.
"""
    chemical = np.asarray(chem, dtype=np.float64)
    available = np.asarray(mask)
    names = np.asarray(ids)
    if isinstance(train_ids, (str, bytes)):
        raise ValueError("Training chemical identities must be a sequence, not a string")
    requested = list(map(str, train_ids))
    if chemical.ndim != 2 or not all(chemical.shape):
        raise ValueError("chem must be nonempty [compound, chemical_coordinate]")
    if available.shape != chemical.shape[:1] or available.dtype != bool:
        raise ValueError("Chemical availability must be boolean and compound-aligned")
    if names.shape != chemical.shape[:1]:
        raise ValueError("Chemical identities must be compound-aligned")
    names = list(map(str, names))
    if (not all(names) or len(set(names)) != len(names) or not requested
            or not all(requested) or len(set(requested)) != len(requested)):
        raise ValueError("Dataset and training chemical identities must be nonempty and unique")
    if isinstance(max_anchors, bool) or not isinstance(max_anchors, Integral) or max_anchors < 1:
        raise ValueError("max_anchors must be a positive integer")
    lookup = {unit: i for i, unit in enumerate(names)}
    if not set(requested).issubset(lookup):
        raise ValueError("A training chemical identity is missing")
    rows = np.asarray([lookup[unit] for unit in requested], dtype=int)
    bit_indices, validity, description = _feature_geometry(chemical.shape[1], chemical_metadata)
    fingerprints, usable = _binary_fingerprints(chemical[rows], available[rows], bit_indices, validity)
    # Selection ties are based on identities, never input row order or outcomes.
    candidates, seen = [], set()
    for row in sorted(np.flatnonzero(usable), key=lambda i: requested[i]):
        packed = np.packbits(fingerprints[row]).tobytes()
        if packed not in seen:
            seen.add(packed)
            candidates.append(int(row))
    selected = []
    if candidates:
        bank = fingerprints[candidates].astype(np.float64)
        mass = bank.sum(axis=1)
        distance = np.full(len(bank), np.inf)
        next_row = 0
        for _ in range(min(int(max_anchors), len(bank))):
            selected.append(candidates[next_row])
            intersect = bank @ bank[next_row]
            union = mass + mass[next_row] - intersect
            distance = np.minimum(distance, 1. - intersect / union)
            distance[[candidates.index(row) for row in selected]] = -np.inf
            next_row = int(np.argmax(distance))
    anchor_ids = [requested[row] for row in selected]
    result = dict(schema_version=ANCHOR_SCHEMA_VERSION, chemical_dim=int(chemical.shape[1]),
        fingerprint_indices=bit_indices, validity_index=validity,
        fingerprint_metadata=description, train_ids=requested,
        training_available=usable.tolist(), training_fingerprints=fingerprints.tolist(),
        anchor_ids=anchor_ids, anchor_fingerprints=fingerprints[selected].tolist(),
        max_anchors=int(max_anchors), unique_training_fingerprints=len(candidates),
        fit_policy="training_structures_only_no_measurements_or_labels",
        selection="farthest_first_1_minus_binary_Tanimoto_start_and_ties_lexicographic_id",
        descriptor_names=["tanimoto_to:" + unit for unit in anchor_ids] + ["fingerprint_bit_density"],
        structured_basis="constant; similarity; similarity_squared; similarity_fourth_power",
        mechanism_scope="chemical_structural_locality_not_verified_target_pathway_or_biological_law")
    validate_anchor_data(result)
    return result


def validate_anchor_data(data):
    """Validate JSON geometry and the identity of every saved training anchor."""
    if not isinstance(data, dict) or data.get("schema_version") != ANCHOR_SCHEMA_VERSION:
        raise ValueError("Unsupported chemical anchor schema")
    required = {"chemical_dim", "fingerprint_indices", "validity_index", "fingerprint_metadata",
        "train_ids", "training_available", "training_fingerprints", "anchor_ids", "anchor_fingerprints",
        "max_anchors", "unique_training_fingerprints", "fit_policy", "selection", "descriptor_names",
        "structured_basis", "mechanism_scope", "schema_version"}
    if set(data) != required:
        raise ValueError("Chemical anchor metadata fields are incomplete or unknown")
    dimension = data["chemical_dim"]
    if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 1:
        raise ValueError("Invalid chemical anchor coordinate count")
    indices, validity, _ = _feature_geometry(dimension, data)
    if indices != data["fingerprint_indices"] or validity != data["validity_index"]:
        raise ValueError("Chemical anchor coordinate declaration changed")
    fingerprint_metadata = data["fingerprint_metadata"]
    if (not isinstance(fingerprint_metadata, dict) or fingerprint_metadata.get("bits") != len(indices)
            or not isinstance(fingerprint_metadata.get("kind"), str)):
        raise ValueError("Fingerprint provenance must match the declared bit coordinates")
    train, anchor = data["train_ids"], data["anchor_ids"]
    if (not isinstance(train, list) or not train or not all(isinstance(x, str) and x for x in train)
            or len(set(train)) != len(train) or not isinstance(anchor, list)
            or not all(isinstance(x, str) and x for x in anchor) or len(set(anchor)) != len(anchor)
            or not set(anchor).issubset(train)):
        raise ValueError("Anchors must have unique training chemical identities")
    mask = np.asarray(data["training_available"])
    fingerprints = np.asarray(data["training_fingerprints"])
    a = np.asarray(data["anchor_fingerprints"])
    if not anchor:
        a = a.reshape(0, len(indices))
    if (mask.shape != (len(train),) or mask.dtype != bool
            or fingerprints.shape != (len(train), len(indices)) or a.shape != (len(anchor), len(indices))
            or not np.isfinite(fingerprints).all() or not np.isfinite(a).all()
            or np.any((fingerprints != 0) & (fingerprints != 1)) or np.any((a != 0) & (a != 1))):
        raise ValueError("Saved chemical fingerprints/masks have invalid geometry or nonbinary values")
    if np.any(fingerprints[~mask]) or np.any(fingerprints[mask].sum(1) == 0):
        raise ValueError("Unavailable chemical structures cannot have active fingerprint payloads")
    lookup = {unit: i for i, unit in enumerate(train)}
    for i, unit in enumerate(anchor):
        if not mask[lookup[unit]] or not np.array_equal(a[i], fingerprints[lookup[unit]]):
            raise ValueError("A saved anchor differs from its available training structure")
    unique = len({np.packbits(row.astype(np.uint8)).tobytes() for row in fingerprints[mask]})
    limit = data["max_anchors"]
    if (isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1
            or data["unique_training_fingerprints"] != unique or len(anchor) != min(limit, unique)
            or len({np.packbits(row.astype(np.uint8)).tobytes() for row in a}) != len(anchor)):
        raise ValueError("Saved chemical landmark count or uniqueness is inconsistent")
    expected = ["tanimoto_to:" + unit for unit in anchor] + ["fingerprint_bit_density"]
    if data["descriptor_names"] != expected or data["fit_policy"] != "training_structures_only_no_measurements_or_labels":
        raise ValueError("Chemical descriptors must be bound to the declared training anchors")
    return data


class ChemistryResponseKernelPrior(nn.Module):
    """Optional KAN-mixed chemical-locality residual on an existing full prior.

All modes see exactly the same descriptor vector and existing chemical token.
``structured`` uses [1,T,T²,T⁴], ``generic`` uses learned Gaussian RBFs, and
``mlp`` uses their common dense response path alone. These are architecture
ablations, not matched-parameter claims. No mode ingests future profiles.
"""
    def __init__(self, chemical_dim, hidden_dim, rank, mode="structured", anchor_data=None):
        super().__init__()
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 1 for x in (chemical_dim, hidden_dim, rank)):
            raise ValueError("Chemical kernel dimensions must be positive integers")
        if mode not in {"structured", "generic", "mlp"}:
            raise ValueError("Chemical kernel mode must be structured, generic or mlp")
        validate_anchor_data(anchor_data)
        if chemical_dim != anchor_data["chemical_dim"]:
            raise ValueError("Chemical kernel and anchor coordinate counts differ")
        self.chemical_dim, self.hidden_dim, self.rank = int(chemical_dim), int(hidden_dim), int(rank)
        self.mode = mode
        self.anchor_data = copy.deepcopy(anchor_data)
        self.validity_index = anchor_data["validity_index"]
        indices = anchor_data["fingerprint_indices"]
        self.register_buffer("fingerprint_indices", torch.tensor(indices, dtype=torch.long))
        anchors = np.asarray(anchor_data["anchor_fingerprints"], dtype=np.float32).reshape(-1, len(indices))
        self.register_buffer("anchor_fingerprints", torch.from_numpy(anchors))
        self.anchor_count = len(anchors)
        self.descriptor_dim, self.basis_size = self.anchor_count + 1, 1 + 3 * self.anchor_count
        self.residual = nn.Sequential(nn.Linear(hidden_dim + self.descriptor_dim, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, hidden_dim))
        if mode != "mlp":
            self.coefficients = nn.Sequential(SplineKANLinear(self.descriptor_dim, 24), nn.Tanh(),
                                               SplineKANLinear(24, self.basis_size))
            self.token_coefficients = nn.Linear(hidden_dim, self.basis_size)
            self.lift = nn.Linear(self.basis_size, hidden_dim, bias=False)
            if mode == "generic":
                self.centers = nn.Parameter(torch.rand(self.basis_size, self.descriptor_dim))
                self.log_width = nn.Parameter(torch.zeros(self.basis_size))
        self.token_delta = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mean_delta = nn.Linear(hidden_dim, rank, bias=False)
        self.log_variance_delta = nn.Linear(hidden_dim, rank, bias=False)
        for layer in (self.token_delta, self.mean_delta, self.log_variance_delta):
            nn.init.normal_(layer.weight, std=.01 / math.sqrt(hidden_dim))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        for name in ("fingerprint_indices", "anchor_fingerprints"):
            key = prefix + name
            if key in state_dict and (state_dict[key].shape != getattr(self, name).shape or not torch.equal(
                    state_dict[key].detach().cpu(), getattr(self, name).detach().cpu())):
                error_msgs.append("Checkpoint chemical anchor geometry differs from its declared metadata: " + key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def descriptors(self, chem, mask):
        if chem.ndim != 2 or chem.shape[1] != self.chemical_dim or not chem.is_floating_point():
            raise ValueError("Chemical kernel requires floating [batch, chemical_coordinate] inputs")
        if mask.shape != chem.shape[:1] or mask.dtype != torch.bool:
            raise ValueError("Chemical kernel mask must be boolean and compound-aligned")
        active = mask.clone()
        if self.validity_index is not None:
            flag = chem[:, self.validity_index]
            if torch.any(active & (~torch.isfinite(flag) | ((flag != 0) & (flag != 1)))):
                raise ValueError("Available structure validity indicators must be binary")
            active = active & (flag == 1)
        bits = chem.index_select(-1, self.fingerprint_indices)
        if torch.any(~torch.isfinite(bits[active])) or torch.any((bits[active] != 0) & (bits[active] != 1)):
            raise ValueError("Available chemical fingerprint coordinates must be binary and finite")
        bits = torch.where(active[:, None], bits, 0.)
        mass = bits.sum(-1)
        active = active & (mass > 0) & (self.anchor_count > 0)
        anchors = self.anchor_fingerprints.to(dtype=chem.dtype)
        intersection = bits @ anchors.T
        union = mass[:, None] + anchors.sum(-1)[None] - intersection
        similarity = intersection / union.clamp_min(1.)
        descriptors = torch.cat((similarity, (mass / len(self.fingerprint_indices))[:, None]), -1)
        return torch.where(active[:, None], descriptors, 0.), active

    def basis(self, descriptors):
        if descriptors.shape[-1] != self.descriptor_dim:
            raise ValueError("Chemical kernel descriptor count changed")
        if self.mode == "mlp":
            raise ValueError("The MLP chemical ablation has no explicit basis")
        if self.mode == "structured":
            similarity = descriptors[..., :self.anchor_count]
            return torch.cat((torch.ones_like(descriptors[..., :1]), similarity,
                              similarity.square(), similarity.pow(4)), -1)
        delta = descriptors.unsqueeze(-2) - self.centers
        width = F.softplus(self.log_width) + 1e-3
        return torch.exp(-.5 * delta.square().mean(-1) / width.square())

    def forward(self, chem, mask, token, mean, variance):
        if (token.shape != (len(chem), self.hidden_dim) or mean.shape != (len(chem), self.rank)
                or variance.shape != mean.shape):
            raise ValueError("Chemical kernel prior/token shapes do not match the configured model")
        if (not torch.isfinite(token).all() or not torch.isfinite(mean).all()
                or not torch.isfinite(variance).all() or torch.any(variance <= 0)):
            raise ValueError("Existing chemical token and positive Gaussian prior must be finite")
        descriptors, active = self.descriptors(chem, mask)
        if not active.any():
            return token, mean, variance
        safe_token = torch.where(active[:, None], token, 0.)
        response = self.residual(torch.cat((safe_token, descriptors), -1))
        if self.mode != "mlp":
            coefficients = torch.tanh(self.coefficients(descriptors) + self.token_coefficients(safe_token))
            coefficients = coefficients / (1. + coefficients.abs().sum(-1, keepdim=True))
            response = response + self.lift(coefficients * self.basis(descriptors))
        updated_token = token + self.token_delta(response)
        updated_mean = mean + self.mean_delta(response)
        log_residual = 2. * torch.tanh(self.log_variance_delta(response) / 2.)
        updated_variance = variance * torch.exp(log_residual)
        if not torch.isfinite(updated_variance[active]).all():
            raise FloatingPointError("Chemical kernel prior variance overflowed")
        return (torch.where(active[:, None], updated_token, token),
                torch.where(active[:, None], updated_mean, mean),
                torch.where(active[:, None], updated_variance, variance))

    def description(self):
        return dict(mode=self.mode, descriptor_names=self.anchor_data["descriptor_names"],
            basis=(self.anchor_data["structured_basis"] if self.mode == "structured" else
                   "learned Gaussian RBF on identical descriptors" if self.mode == "generic" else
                   "dense MLP on identical descriptors and chemical token"),
            mixing="cubic B-spline KAN and chemical-token coefficients" if self.mode != "mlp" else "dense MLP",
            anchor_count=self.anchor_count, basis_size=None if self.mode == "mlp" else self.basis_size,
            prior_conditioning="chemical structure only; no current or future measurement",
            claims="soft chemical-locality prior; not verified targets, pathways or a biological law")
