"""Typed, provenance-bearing biological annotations and an optional latent prior.

No morphology coordinate is interpreted as a gene. Only declared, decision-time
input relations enter the train-fitted relation encoder under an explicit support
weight policy. Supplied confidence and policy-assigned unit support remain
distinct; neither is a calibrated probability of a cellular effect. Missing and
out-of-vocabulary mechanisms preserve the base prior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

BIOLOGY_SCHEMA_VERSION = 1
TOKEN_FIELDS = ("subject", "object", "predicate", "direction", "evidence_family", "evidence_code")
GO_EXPERIMENTAL = frozenset("EXP IDA IPI IMP IGI IEP HTP HDA HMP HGI HEP".split())
GO_FAMILIES = {**dict.fromkeys(GO_EXPERIMENTAL, "experimental"),
    **dict.fromkeys("IBA IBD IKR IRD".split(), "phylogenetic"),
    **dict.fromkeys("ISS ISO ISA ISM IGC RCA".split(), "computational"),
    **dict.fromkeys("TAS NAS".split(), "author"), **dict.fromkeys("IC ND".split(), "curatorial"),
    "IEA": "automatic"}
WEIGHT_POLICIES = {"supplied_confidence_only", "confidence_or_unit_support"}
PERTURBATION_TYPES = frozenset({"small_molecule", "gene_knockdown", "gene_knockout",
    "gene_activation", "gene_overexpression", "cytokine", "combination", "vehicle_control",
    "negative_control", "other", "unknown"})


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _entity(value, name="entity"):
    _text(value, name)
    if ":" not in value or not all(value.split(":", 1)):
        raise ValueError(f"{name} needs a namespace, e.g. ENSEMBL:ENSG00000141510")
    return value


@dataclass(frozen=True)
class Provenance:
    source: str
    reference: str
    source_version: str
    available_at: str

    def __post_init__(self):
        for name in ("source", "reference", "source_version", "available_at"):
            _text(getattr(self, name), name)


@dataclass(frozen=True)
class TypedValue:
    """Observed, planned and nominal quantities remain distinct declarations."""
    kind: str
    status: str
    value: Any = None
    unit: str | None = None
    provenance: Provenance | None = None
    role: str = "input"
    availability: str = "unknown"
    interpretation: str = "reported"

    def __post_init__(self):
        if isinstance(self.provenance, dict):
            object.__setattr__(self, "provenance", Provenance(**self.provenance))
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise ValueError("Metadata provenance must be a typed source declaration")
        if self.kind not in {"category", "quantity", "entities"}:
            raise ValueError("Typed metadata kind must be category, quantity or entities")
        if self.status not in {"known", "unknown", "not_applicable"}:
            raise ValueError("Explicit known/unknown/not_applicable metadata status required")
        if self.role not in {"input", "validation", "audit"} or self.availability not in {"decision", "after_measurement", "unknown"}:
            raise ValueError("Declare metadata role and information availability")
        if self.interpretation not in {"reported", "nominal_protocol", "planned", "observed"}:
            raise ValueError("Metadata interpretation must distinguish nominal/planned/observed")
        if self.status != "known":
            if self.value is not None:
                raise ValueError("Unknown/not-applicable metadata cannot carry an invented value")
            return
        if self.provenance is None:
            raise ValueError("Known biological metadata requires provenance")
        if self.kind == "quantity":
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
                raise ValueError("A known quantity must be finite numeric data")
            _text(self.unit, "quantity unit")
        elif self.kind == "category":
            _text(self.value, "category value")
            if self.unit is not None:
                raise ValueError("Categories do not have quantity units")
        else:
            if not isinstance(self.value, (list, tuple)) or not self.value:
                raise ValueError("Known entities require a nonempty namespaced list")
            for value in self.value:
                _entity(value)
            object.__setattr__(self, "value", tuple(self.value))
            if self.unit is not None:
                raise ValueError("Entity sets do not have quantity units")


@dataclass(frozen=True)
class MechanismRelation:
    subject: str
    predicate: str
    object: str
    direction: str
    evidence_family: str
    evidence_code: str
    confidence: float | None
    provenance: Provenance
    role: str = "input"
    availability: str = "unknown"

    def __post_init__(self):
        if isinstance(self.provenance, dict):
            object.__setattr__(self, "provenance", Provenance(**self.provenance))
        if not isinstance(self.provenance, Provenance):
            raise ValueError("Every relation needs source provenance")
        _entity(self.subject, "relation subject")
        _entity(self.object, "relation object")
        _text(self.predicate, "relation predicate")
        _text(self.evidence_code, "evidence code")
        if self.direction not in {"activation", "inhibition", "association", "membership", "unknown"}:
            raise ValueError("Declare activation/inhibition/association/membership/unknown direction")
        if self.evidence_family == "curated":
            object.__setattr__(self, "evidence_family", "curatorial")
        elif self.evidence_family == "author_statement":
            object.__setattr__(self, "evidence_family", "author")
        if self.evidence_family not in {"experimental", "phylogenetic", "computational", "author", "curatorial", "automatic", "unknown"}:
            raise ValueError("Unknown biological evidence family")
        if self.evidence_code in GO_FAMILIES and self.evidence_family != GO_FAMILIES[self.evidence_code]:
            raise ValueError("GO evidence code has an inconsistent evidence family")
        if self.confidence is not None and (isinstance(self.confidence, bool) or
                not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence) or
                not 0 <= self.confidence <= 1):
            raise ValueError("Confidence must be supplied in [0,1], or explicitly unknown")
        if self.role not in {"input", "validation", "audit"} or self.availability not in {"decision", "after_measurement", "unknown"}:
            raise ValueError("Declare relation role and decision-time availability")

    @property
    def usable(self):
        return self.support_weight("confidence_or_unit_support") > 0

    def support_weight(self, policy):
        if policy not in WEIGHT_POLICIES:
            raise ValueError("Declare a biological evidence-weight policy")
        if self.role != "input" or self.availability != "decision":
            return 0.
        if self.evidence_code == "ND":
            # GO ND says no biological data are available; it is not a
            # positive mechanism assertion even if a source supplies a score.
            return 0.
        if self.confidence is not None:
            return float(self.confidence)
        # Unit support is an explicit model convention, never an invented
        # confidence of 1. Evidence family/code remain separate learned inputs.
        return float(policy == "confidence_or_unit_support" and self.evidence_family != "unknown")

    @property
    def tokens(self):
        return tuple(getattr(self, name) for name in TOKEN_FIELDS)


@dataclass(frozen=True)
class BiologyRecord:
    unit_id: str
    perturbation_type: str = "unknown"
    perturbation: Mapping[str, TypedValue] = field(default_factory=dict)
    biological_context: Mapping[str, TypedValue] = field(default_factory=dict)
    measurement_metadata: Mapping[str, Mapping[str, TypedValue]] = field(default_factory=dict)
    relation_coverage: str = "unknown"
    coverage_provenance: Provenance | None = None
    relations: tuple[MechanismRelation, ...] = ()

    def __post_init__(self):
        _text(self.unit_id, "unit_id")
        if self.perturbation_type not in PERTURBATION_TYPES:
            raise ValueError("Unrecognized perturbation type")
        for name in ("perturbation", "biological_context"):
            values = {str(k): v if isinstance(v, TypedValue) else TypedValue(**v) for k, v in getattr(self, name).items()}
            object.__setattr__(self, name, values)
        object.__setattr__(self, "measurement_metadata", {
            str(well): {str(k): v if isinstance(v, TypedValue) else TypedValue(**v) for k, v in values.items()}
            for well, values in self.measurement_metadata.items()})
        relations = tuple(r if isinstance(r, MechanismRelation) else MechanismRelation(**r) for r in self.relations)
        object.__setattr__(self, "relations", relations)
        if isinstance(self.coverage_provenance, dict):
            object.__setattr__(self, "coverage_provenance", Provenance(**self.coverage_provenance))
        if self.coverage_provenance is not None and not isinstance(self.coverage_provenance, Provenance):
            raise ValueError("Coverage provenance must be a typed source declaration")
        if self.relation_coverage not in {"unknown", "partial", "complete", "known_empty", "not_applicable"}:
            raise ValueError("Relation coverage must be explicit")
        if self.relation_coverage in {"unknown", "known_empty", "not_applicable"} and any(r.role == "input" for r in relations):
            raise ValueError("Declared input relations require partial or complete coverage")
        if self.relation_coverage in {"known_empty", "not_applicable"}:
            if self.perturbation_type not in {"vehicle_control", "negative_control"} or self.coverage_provenance is None:
                raise ValueError("Known-empty/no-target declarations require a documented control, not missing drug annotation")
        if self.relation_coverage == "complete" and self.coverage_provenance is None:
            raise ValueError("Complete knowledge coverage requires a source declaration")
        root = self.perturbation.get("entities")
        roots = set(root.value) if root and root.kind == "entities" and root.status == "known" and root.role == "input" and root.availability == "decision" else set()
        seen, roles = set(), {}
        for relation in relations:
            # An annotation used as an input cannot also be an independent
            # biological validation label, even under another source spelling.
            subject = "PERTURBATION:self" if relation.subject in roots else relation.subject
            key = (subject, relation.predicate, relation.object)
            roles.setdefault(key, set()).add(relation.role)
            if {"input", "validation"}.issubset(roles[key]):
                raise ValueError("The same relation cannot be input and independent validation")
            serialized = json.dumps(asdict(relation), sort_keys=True)
            if serialized in seen:
                raise ValueError("Duplicate relations would duplicate evidence weight")
            seen.add(serialized)
        reachable = roots | {"PERTURBATION:self"}
        remaining = [r for r in relations if r.role == "input"]
        while remaining:
            linked = [r for r in remaining if r.subject in reachable]
            if not linked:
                raise ValueError("Input relation subjects must connect to declared perturbation entities or PERTURBATION:self")
            reachable.update(r.object for r in linked)
            remaining = [r for r in remaining if r not in linked]

    def to_dict(self):
        return asdict(self)


def validate_records(records, ids, well_ids=None):
    parsed = tuple(record if isinstance(record, BiologyRecord) else BiologyRecord(**record) for record in records)
    expected = list(map(str, ids))
    lookup = {record.unit_id: record for record in parsed}
    if len(lookup) != len(parsed) or set(lookup) != set(expected):
        raise ValueError("Biology records must match every dataset unit exactly once")
    result = tuple(lookup[unit] for unit in expected)
    if well_ids is not None:
        for record, wells in zip(result, well_ids):
            if set(record.measurement_metadata) - set(map(str, wells)):
                raise ValueError("Measurement metadata references another unit's physical well")
    return result


def load_biology(path):
    value = json.loads(Path(path).read_text())
    if set(value) != {"schema_version", "records"} or value["schema_version"] != BIOLOGY_SCHEMA_VERSION:
        raise ValueError("Unsupported biological annotation schema")
    return tuple(BiologyRecord(**record) for record in value["records"])


def save_biology(records, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Biological metadata destination already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": BIOLOGY_SCHEMA_VERSION,
        "records": [r.to_dict() for r in records]}, indent=2, allow_nan=False) + "\n")


def _relation_tokens(record, relation):
    root = record.perturbation.get("entities")
    roots = set(root.value) if root and root.kind == "entities" and root.status == "known" and root.role == "input" and root.availability == "decision" else set()
    subject = "PERTURBATION:self" if relation.subject in roots else relation.subject
    return (subject,) + tuple(getattr(relation, name) for name in TOKEN_FIELDS[1:])


def _reachable_relations(record, relations):
    """Only admitted, available paths can unlock downstream mechanism edges."""
    root = record.perturbation.get("entities")
    roots = set(root.value) if root and root.kind == "entities" and root.status == "known" and root.role == "input" and root.availability == "decision" else set()
    reachable, remaining, admitted = roots | {"PERTURBATION:self"}, list(relations), []
    while remaining:
        linked = [r for r in remaining if r.subject in reachable]
        if not linked:
            break
        admitted.extend(linked)
        reachable.update(r.object for r in linked)
        remaining = [r for r in remaining if r not in linked]
    return admitted


def fit_vocabulary(records, train_ids, evidence_weight_policy="confidence_or_unit_support"):
    """No vocabulary entry or truncation limit is learned from held-out units."""
    train_ids = list(map(str, train_ids))
    if not train_ids or len(train_ids) != len(set(train_ids)):
        raise ValueError("Vocabulary requires unique nonempty training identities")
    lookup = {r.unit_id: r for r in records}
    if set(train_ids) - set(lookup):
        raise ValueError("Missing training biological record")
    if evidence_weight_policy not in WEIGHT_POLICIES:
        raise ValueError("Unknown evidence-weight policy")
    usable = [_relation_tokens(lookup[unit], r) for unit in train_ids
              for r in _reachable_relations(lookup[unit], [r for r in lookup[unit].relations
                  if r.support_weight(evidence_weight_policy) > 0])]
    return {"schema_version": 1, "fit_policy": "training_units_only",
            "evidence_weight_policy": evidence_weight_policy, "train_ids": train_ids, "fields": {
                name: sorted({tokens[j] for tokens in usable}) for j, name in enumerate(TOKEN_FIELDS)}}


def validate_vocabulary(value):
    if not isinstance(value, dict) or set(value) != {"schema_version", "fit_policy", "evidence_weight_policy", "train_ids", "fields"}:
        raise ValueError("Malformed biological vocabulary")
    if value["schema_version"] != 1 or value["fit_policy"] != "training_units_only":
        raise ValueError("Only explicit training-unit biological vocabularies are supported")
    if value["evidence_weight_policy"] not in WEIGHT_POLICIES:
        raise ValueError("Unknown evidence-weight policy in biological vocabulary")
    if not isinstance(value["train_ids"], list) or not value["train_ids"] or any(not isinstance(x, str) or not x for x in value["train_ids"]) or len(set(value["train_ids"])) != len(value["train_ids"]):
        raise ValueError("Vocabulary fitting identities must be nonempty and unique")
    if set(value["fields"]) != set(TOKEN_FIELDS):
        raise ValueError("Vocabulary semantic fields differ")
    for values in value["fields"].values():
        if not isinstance(values, list) or values != sorted(set(values)) or any(not isinstance(x, str) or not x for x in values):
            raise ValueError("Vocabulary fields must be sorted, unique strings")
    return value


def encode_relations(records, vocabulary):
    """Validation-only/post-outcome annotations are excluded before tensorization."""
    validate_vocabulary(vocabulary)
    policy = vocabulary["evidence_weight_policy"]
    rows, oov_counts, unreachable_counts = [], [], []
    lookups = [{value: i + 1 for i, value in enumerate(vocabulary["fields"][field])} for field in TOKEN_FIELDS]
    for record in records:
        candidates = [r for r in record.relations if r.support_weight(policy) > 0]
        reachable = _reachable_relations(record, candidates)
        known = [r for r in reachable if all(lookup.get(value, 0)
                 for lookup, value in zip(lookups, _relation_tokens(record, r)))]
        admitted = _reachable_relations(record, known)
        # Removing an OOV bridge must also remove otherwise-known descendants.
        oov_counts.append(len(reachable) - len(known))
        unreachable_counts.append(len(candidates) - len(reachable) + len(known) - len(admitted))
        unique = {}
        for relation in admitted:
            support = relation.support_weight(policy)
            tokens = _relation_tokens(record, relation)
            previous = unique.get(tokens)
            if previous is None or support > previous[1]:
                unique[tokens] = (relation, support)
        # Keep distinct evidence channels, but one semantic edge receives a
        # total support budget of max(channel supports), never their sum.
        groups = {}
        for tokens, item in sorted(unique.items()):
            groups.setdefault(tokens[:4], []).append((tokens, *item))
        row = []
        for group_id, channels in enumerate(groups.values()):
            budget = max(support for _, _, support in channels)
            total = sum(support for _, _, support in channels)
            row.extend((tokens, relation, budget * (support / total), group_id)
                       for tokens, relation, support in channels)
        rows.append(row)
    length = max(1, max(map(len, rows), default=0))
    token = np.zeros((len(rows), length, len(TOKEN_FIELDS)), np.int64)
    weight = np.zeros((len(rows), length), np.float64)
    mask = np.zeros((len(rows), length), bool)
    confidence = np.zeros((len(rows), length), np.float64)
    confidence_mask = np.zeros((len(rows), length), bool)
    group_ids = np.full((len(rows), length), -1, np.int64)
    for i, relations in enumerate(rows):
        for j, (tokens, relation, support, group_id) in enumerate(relations):
            encoded = [lookup.get(value, 0) for lookup, value in zip(lookups, tokens)]
            token[i, j], weight[i, j], mask[i, j], group_ids[i, j] = encoded, support, True, group_id
            if relation.confidence is not None:
                confidence[i, j], confidence_mask[i, j] = relation.confidence, True
    return {"biology_tokens": token, "biology_support_weight": weight,
            "biology_confidence": confidence, "biology_confidence_mask": confidence_mask,
            "biology_mask": mask, "biology_group_ids": group_ids,
            "biology_oov_count": np.asarray(oov_counts, np.int64),
            "biology_unreachable_count": np.asarray(unreachable_counts, np.int64)}


class EvidenceAwareMechanismPrior(nn.Module):
    """Soft relation-set residual on p(z); not a gene-level mechanistic simulator."""
    def __init__(self, vocabulary, hidden_dim, rank):
        super().__init__()
        validate_vocabulary(vocabulary)
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(vocabulary["fields"][name]) + 1, hidden_dim, padding_idx=0)
            for name in TOKEN_FIELDS])
        self.relation = nn.Sequential(nn.Linear(hidden_dim * len(TOKEN_FIELDS), hidden_dim), nn.GELU())
        self.mean = nn.Linear(hidden_dim, rank, bias=False)
        self.log_variance = nn.Linear(hidden_dim, rank, bias=False)
        # A soft, small initial residual; support weight attenuates rather than
        # cancels in a normalized weighted average of a single relation.
        nn.init.normal_(self.mean.weight, std=.01 / math.sqrt(hidden_dim))
        nn.init.normal_(self.log_variance.weight, std=.01 / math.sqrt(hidden_dim))

    def forward(self, batch, prior_mean, prior_variance):
        if "biology_tokens" not in batch:
            return prior_mean, prior_variance
        tokens, support, mask = batch["biology_tokens"], batch["biology_support_weight"], batch["biology_mask"]
        if tokens.ndim != 3 or tokens.shape[-1] != len(self.embeddings) or mask.shape != tokens.shape[:2] or support.shape != mask.shape or mask.dtype != torch.bool:
            raise ValueError("Invalid biological relation tensors")
        if tokens.dtype != torch.long or tokens.shape[0] != prior_mean.shape[0]:
            raise ValueError("Biological tokens must be batch-aligned long integers")
        if not torch.isfinite(support[mask]).all() or torch.any((support[mask] <= 0) | (support[mask] > 1)):
            raise ValueError("Active biological support weights must be finite in (0,1]")
        for j, embedding in enumerate(self.embeddings):
            if torch.any((tokens[..., j][mask] <= 0) | (tokens[..., j][mask] >= embedding.num_embeddings)):
                raise ValueError("Unknown biological entities cannot be marked available")
        active = mask.any(-1, keepdim=True)
        if not active.any():
            return prior_mean, prior_variance
        groups = batch["biology_group_ids"]
        if groups.shape != mask.shape or groups.dtype != torch.long or torch.any(groups[mask] < 0):
            raise ValueError("Active biological evidence channels require semantic group IDs")
        denominator = []
        for i in range(len(mask)):
            group_ids = torch.unique(groups[i][mask[i]])
            for group in group_ids:
                if support[i][mask[i] & (groups[i] == group)].sum() > 1. + 1e-6:
                    raise ValueError("One semantic relation cannot receive support greater than one")
            denominator.append(max(1, len(group_ids)))
        safe = torch.where(mask[..., None], tokens, 0)
        encoded = self.relation(torch.cat([embedding(safe[..., j]) for j, embedding in enumerate(self.embeddings)], -1))
        weights = torch.where(mask, support, 0).to(encoded.dtype)
        pooled = (encoded * weights[..., None]).sum(1) / encoded.new_tensor(denominator)[:, None]
        mean = prior_mean + self.mean(pooled)
        variance = prior_variance * torch.exp(2. * torch.tanh(self.log_variance(pooled) / 2.))
        return torch.where(active, mean, prior_mean), torch.where(active, variance, prior_variance)
