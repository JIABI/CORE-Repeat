# Optional biological metadata and mechanism prior

This interface adds provenance-bearing biological annotations to the existing
measurement dataset. The optional mechanism prior is a soft correction to the
chemical Gaussian latent prior. It is not a gene-level simulator, a hard
Cell Painting feature-to-gene map, or a replacement for the measurement model.
The observed measurement coordinates, endpoint, actions and existing results are
unchanged. `use_biology_prior` defaults to `false`.

## 1. Sidecar structure

The standalone UTF-8 JSON sidecar has exactly two top-level fields:

```json
{"schema_version": 1, "records": []}
```

There is one record per dataset unit, matched by `unit_id`, not row order. A
record has these fields:

| Field | Meaning |
| --- | --- |
| `unit_id` | Exact dataset unit identifier; never a learned biological entity token. |
| `perturbation_type` | Small molecule, gene knockdown/knockout/activation/overexpression, cytokine, combination, vehicle/negative control, other or unknown. |
| `perturbation` | Named typed fields, such as entities, SMILES, dose, exposure duration and perturbation reagents. |
| `biological_context` | Named typed fields, such as organism, cell line/type, baseline state and culture conditions. |
| `measurement_metadata` | A dictionary keyed by this unit's physical well IDs, containing typed condition/QC fields. |
| `relation_coverage` | `unknown`, `partial`, `complete`, `known_empty` or `not_applicable`. |
| `coverage_provenance` | A source declaration, required for complete or known-empty coverage. |
| `relations` | Directed, evidence-bearing semantic relations, with explicitly separated input, validation and audit roles. |

Missing targets mean `unknown`, not an empty known target set. `known_empty` and
`not_applicable` are restricted to documented vehicle/negative controls in this
implementation. Unannotated drugs must not use these statuses. `partial` is
appropriate for any drug with usable but incomplete target knowledge.

## 2. Typed values, quantities and provenance

Every typed field declares:

```json
{
  "kind": "quantity",
  "status": "known",
  "value": 10.0,
  "unit": "uM",
  "provenance": {
    "source": "dataset protocol",
    "reference": "a stable source URL or local provenance reference",
    "source_version": "an explicit release or revision",
    "available_at": "2026-09-12"
  },
  "role": "input",
  "availability": "decision",
  "interpretation": "nominal_protocol"
}
```

This example means a protocol-specified dose, not a confirmed executed dose.
`interpretation` distinguishes `nominal_protocol`, `planned`, `observed` and
`reported`. Dose requires concentration units; time requires time units; cell
counts can use `cells`. There is no implicit unit conversion or inferred dose law.

The three `kind` values are `category`, `quantity` and `entities`. A known
quantity requires a finite number and units; a known category requires a string;
known entities require a nonempty list of namespaced identifiers. All known
values require provenance. Unknown/not-applicable fields have `value: null`.

`role` is `input`, `validation` or `audit`; `availability` is `decision`,
`after_measurement` or `unknown`. A future observed cell count is not a planned
sampling condition. The typed metadata fields are preserved for audit and future
declared encoders; they are **not automatically copied into neural inputs**.
In this implementation only the admitted mechanism relation set enters the new
prior. Existing condition/reference inputs retain their current information
boundary. Consequently, attaching future-well QC cannot expose it to the model.

## 3. Relations and root semantics

An input drug-target relation can be declared as:

```json
{
  "subject": "PERTURBATION:self",
  "predicate": "targets",
  "object": "HGNC:example_identifier",
  "direction": "inhibition",
  "evidence_family": "curatorial",
  "evidence_code": "not_supplied",
  "confidence": null,
  "provenance": {
    "source": "curated annotation source",
    "reference": "a stable evidence reference",
    "source_version": "declared release",
    "available_at": "2026-09-12"
  },
  "role": "input",
  "availability": "decision"
}
```

These are schema placeholders, not biological assertions about any real entity.
`direction` is activation, inhibition, association, membership or unknown.
Unknown direction is not converted to inhibition. An ambiguous chemical target
or mechanism should remain `audit` until reviewed.

Alternatively, the root subject can be one of the declared decision-time input
entities in `perturbation.entities`. The encoder normalizes that own-perturbation
subject to `PERTURBATION:self`, avoiding a separate learned drug-ID token. A
target-to-pathway edge retains its target subject. Every input edge must be
reachable from a declared root or the sentinel, so an unrelated gene graph cannot
silently become a drug's mechanism. Biological effects are not deduced from this
reachability constraint; it only validates graph attribution.

Both subject and object, relation type, direction, evidence family and evidence
code are encoded. Multiple source copies of an identical statement retain their
provenance in the sidecar but do not duplicate model evidence. Distinct evidence
families/codes for the same subject/object/predicate/direction remain distinct
channels, sharing a total support budget equal to the largest admitted channel
weight. Pooling is normalized by semantic edges, not evidence channels. Thus an
additional database alias does not amplify an edge and an additional evidence
channel does not dilute its weight relative to other semantic edges. The same subject/predicate/object
cannot be declared both input and independent-validation evidence. This check
does not establish biological independence between differently named labels;
any external validation must also exclude overlapping target/pathway knowledge
by study design.

The evidence families follow the GO distinction:

| Family | Recognized GO codes |
| --- | --- |
| experimental | EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP |
| phylogenetic | IBA, IBD, IKR, IRD |
| computational | ISS, ISO, ISA, ISM, IGC, RCA |
| author | TAS, NAS |
| curatorial | IC, ND |
| automatic | IEA |
| unknown | An explicitly unavailable evidence classification |

Recognized code/family mismatches are rejected. Non-GO evidence codes can be
preserved verbatim with their declared family. `curated` is accepted as an alias
for `curatorial`. No evidence code is converted to a probability. `ND` means no
biological data are available and is never admitted as positive mechanism
support. A GO `NOT`/negative annotation requires explicitly reviewed negative
relation semantics; it must not be imported as an ordinary positive membership
edge. This interface does not automatically import GO rows or infer qualifiers.

## 4. Optional prior and support policy

Enable explicitly in the training configuration:

```json
{
  "use_biology_prior": true,
  "biology_evidence_weight_policy": "confidence_or_unit_support"
}
```

The two support policies are:

1. `confidence_or_unit_support`: supplied numeric confidence in (0, 1] is used
   as a soft weight; if confidence is absent but the evidence family is declared,
   the relation receives unit **support**, an explicit modeling convention.
2. `supplied_confidence_only`: only positive supplied confidence is admitted.

Under the first policy a curated edge with missing confidence has
`biology_support_weight=1`, `biology_confidence=0` and
`biology_confidence_mask=false`. This is not a claim of confidence one. Unknown
evidence with no supplied confidence, zero-confidence statements, validation/audit
relations and non-decision-time statements do not enter the model.

For each admitted evidence channel, the six semantic embeddings are concatenated
and transformed. Channels for a semantic edge share its support budget in
proportion to their declared weights. The resulting weighted representations are
summed and divided by the number of admitted semantic edges, so reducing
confidence attenuates even a single-edge prior. Two learned residual heads adjust
the existing prior:

```text
z_mean = chemical_prior_mean + biological_mean_residual
z_variance = chemical_prior_variance * exp(bounded_log_variance_residual)
```

The log-variance residual is bounded to (-2, 2). This is numerical regularization,
not a biological law. The prior remains Gaussian and the original conditional
measurement model handles repeated measurements without changing the likelihood
space. Source/evidence semantics can affect the learned residual; all source
provenance remains in the serialized annotations.

## 5. Vocabulary, missingness and checkpoints

`TrainScaler.fit(..., biology_enabled=True)` fits the semantic vocabulary only on
the declared training units and admitted input relations. It records these unit
IDs and the support policy. A held-out unit with known train-vocabulary relations
can use the prior; an edge with any out-of-vocabulary semantic token is masked as
a whole and counted in `biology_oov_count`. It never activates an untrained random
embedding. Availability and connectivity are rechecked after the support policy
and after OOV filtering: an unavailable, zero-support or OOV drug-target bridge
cannot unlock a target-pathway edge. Blocked descendants are counted in
`biology_unreachable_count`. OOV/unreachable relations are not equivalent to
biologically absent mechanisms.

The six token fields, supplied-confidence values/masks, per-channel support
weights, semantic group IDs and availability mask pass through inference batches and variable-length episode
collation. Validation-only and audit-only edges are excluded before tensorization,
including vocabulary construction and sequence length.

The vocabulary is retained in the scaler and model checkpoint, with configuration
and training-unit consistency checks at reload. Resume binds the active input
relations and perturbation-root declarations; editing these inputs cannot silently
resume a run. Old default-off checkpoints without any biological keys load with
the original architecture. The optional module is initialized without changing
the original modules' weights or global RNG state. When disabled or when all
relations are unknown/masked, predictions exactly follow the original prior.

## 6. Attach metadata without training

Create and validate a standalone sidecar with `BiologyRecord`, `TypedValue`,
`MechanismRelation`, `Provenance`, `validate_records` and `save_biology` from
`opal2.biology`. Then attach it to a **new** portable dataset export:

```sh
.venv/bin/python -m opal2.cli attach-biology \
  --dataset /absolute/path/existing/measurements.npz \
  --annotations /absolute/path/biology.json \
  --output /absolute/path/new/measurements.npz
```

Existing destinations are rejected. This command does not fit a vocabulary,
train a model or change split assignments. It preserves all measurement arrays;
the new JSON dataset sidecar stores the typed annotations. An experiment using
the new export must keep its existing declared split file and reference policy.

`tests/test_biology_prior.py` exercises this path with engineering fixtures,
including actual forward/gradient effects, all-unknown fallback, old checkpoint
compatibility and the exclusion of future/validation annotations. These tests do
not constitute biological model validation or improved experimental results.
