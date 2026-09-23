# Measurement data, reference identity, and decision-time information

R2 portable exports use schema version 2: compressed NPZ arrays without pickled
objects plus a JSON metadata sidecar. Version-1 files remain readable when their
physical identities are explicit. Measurements remain fixed observed feature
coordinates, not learned embeddings.

## Measurement arrays

| Field | Shape | Meaning |
|---|---|---|
| `Y` | N × W × D | Observed profiles; absent outcomes may be NaN |
| `observed_mask`, `well_mask` | N × W | Outcome existence and planned slot existence, respectively |
| `ids`, `well_ids` | N; N × W | Unique compound IDs and globally unique physical well IDs |
| `feature_names`, `feature_group_index` | D | Fixed coordinate identity and named group assignment |
| `cond` | N × W × K | Declared pre-measurement conditions; no executed future QC |
| `reference`, `reference_mask` | N × W × 3 × R; N × W × 3 | Genuine reference summaries and availability at source/batch/plate tiers |
| `groups` | N × W × 3 | Source/batch/plate identities for covariance sharing |
| `chem`, `chem_mask` | N × H; N | Chemical descriptors and availability; missing structures are zeroed and separately masked |
| `n_cells`, `n_cells_mask` | N × W | Optional observed cell counts; never automatically exposed for future wells |

Non-integral indices are rejected, not truncated. Physical well IDs cannot be
duplicated across compounds, roles, or repeated fields of view. Synthetic IDs
are permitted only for explicitly marked software fixtures, and are never a
physical-identity audit pass. Empty observed context is supported for prior-only
decisions. Padding and missing chemistry do not become observations.

CellProfiler groups preserve compartment, feature family, and recognized channel
tokens (DNA, RNA, ER, AGP, Mito, and related explicitly named channels). A feature
family is not asserted to be a biological pathway.

## Compact identity-matched reference catalog

| Field | Shape | Meaning |
|---|---|---|
| `panel_y` | M × D | Real reference profiles in the same fixed coordinate system |
| `panel_ids`, `panel_identity` | M | Unique catalog ID and reference compound/control identity |
| `panel_groups` | M × 3 | Actual source/batch/plate ownership |
| `panel_members` | M × Q | Physical constituent control IDs, empty-string padding |
| `panel_index` | N × W × 3 × P | Catalog indices for each declared reference tier; −1 padding |
| `panel_template`, `panel_template_mask` | M × D; M | Train-fitted, identity-matched typical reference profiles and availability |

The same physical control is stored once; different hierarchy memberships reuse
its catalog index. Catalog rows may not duplicate physical controls or overlap
treatment outcomes. Source-5 R2 exports retain every available individual DMSO
control, not an averaged pseudo-well. There is no invented Target-2 panel.

`TrainScaler.fit(dataset, train_indices)` fits coordinate transforms and reference
identity templates only from available reference rows owned by training wells.
It saves identity means/counts and the fitting panel IDs, not a duplicated dense
control matrix. Templates for fitting controls explicitly exclude that control;
a lone training reference has no leave-one-reference-out template. Novel
reference identities remain unanchored and are masked. When applying a saved
scaler, templates supplied in a new data file are not trusted or refitted.

Inference emits a compact accessible catalog:

- `panel_catalog_y`, `panel_catalog_template`: M_visible × D;
- `panel_catalog_template_mask`, `panel_catalog_ids`: M_visible;
- `context_panel_index`, `target_panel_index`: B × C/T × 3 × P;
- matching panel, identity, and template-availability masks.

The catalog contains only controls allowed for that decision. It is not repeated
over B × C × P. Episode collation unions catalog identities and remaps each local
index before model encoding. A shared control is one measurement, not many
independent training examples.

## Legal library context

`fit_library_context(normalized_dataset, train_indices, context_index=0,
max_neighbors=32)` creates a `LibraryBank` from the declared initial training well
only. It serializes to NPZ via `bank.save()` and reloads with `LibraryBank.load()`.
`attach_library_context(dataset, bank)` checks the coordinate transform and
feature order before attaching it. No target outcome is consulted.

For each query the bank excludes the same compound. Neighbor retrieval gives
same-plate controls priority, followed by same-source and other-source candidates,
using observed-profile squared distance and stable IDs to break ties. It returns
up to the explicitly configured number of neighbors; it also supplies full-bank
mean, variance, count, and density statistics. In a zero-well state, retrieval
uses declared conditions rather than reading a hidden initial profile. These
are contextual descriptors, not independent sample counts or causal density
estimates. Validation/evaluation compounds never enter the training bank.

## Availability is distinct from normalization

The historical endpoint uses controls on each measured outcome's plate. This
does not grant those future controls to the predictor. Default
`reference_access="observed_only"` permits context-associated references under
the stated retrospective timing assumption and hides every target reference.
`none` hides all reference inputs. `all_declared` requires a separate declaration
that genuine candidate-condition reference panels are available before action.
File existence alone does not establish historical chronology.

Executed future cell counts remain masked in inference even if present in an
offline file. A future count predictor, if introduced, would be a separately
declared model estimate, not this observed-QC field.

## Training and source separation

`make_inference_batch` reads context outcomes only. `make_episode` and
`make_training_batch` keep `target_y` and its observed mask outside model inputs.
Variable context sizes, including zero, are collated with explicit masks.
Pair expansion creates prediction tasks, not new independent experimental units.

`materialize_source_partitions` first removes all other-source treatment values,
conditions and catalog rows, then compacts the retained wells in their declared
original order. It rejects a reference panel owned by an excluded source. New
source / known compound transfer requires explicit `allow_known_compounds=True`.
`merge_compound_disjoint_partitions` connects genuinely source- and compound-
disjoint objects to the ordinary training interface and returns preserved split
indices. It rejects identity renaming, overlapping compounds, and overlapping
sources. A single-source dataset cannot exercise this transfer evaluation.

The affine measurement transform is training-only and invertible. Utility is
computed after `inverse_y`; no new clipping, dimension selection, or endpoint
redefinition is performed by the model scaler.

## Source-5 R2 data scope

The real-data inputs for R2 are `data/source5_primary_fullcontrols` and
`data/source5_spatial_fullcontrols`. Each retains exactly the previously opened
639 compounds × four assigned X/Z1/Z2/V wells × 3617 endpoint coordinates. The
compound split is unchanged: 383 train, 96 validation, 64 calibration, 96
development evaluation. These are previously explored DEV objects, not FINAL.

Chemical descriptors are actual RDKit Morgan radius-2 fingerprints plus a valid-
SMILES indicator. Planned conditions retain the existing dose/position and known
source/batch/plate descriptors. Full reference profiles use the endpoint's fixed
coordinates, fixed MAD scales, and DMSO plate centering. Spatial reference
correction is fitted only on genuine whole-DMSO controls using the previously
declared row/column model. Reference summaries retain their declared raw-scaled
control basis; they are not mislabeled as outcome profiles.

Source-level references and Target-2 are missing, and actual cell counts are not
available in this authorized export. The adapter neither searches for nor reads
fifth repeats, FINAL profiles, or unopened data. Allocation metadata may be read
only to enforce the existing exclusions. Reusing the existing DMSO reference
measurements is an explicit retrospective availability/cost assumption, not a
new prospective experiment or evidence of cross-source adaptation.
