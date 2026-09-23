# LKCP optional-module condition-transfer comparison

This execution addendum is recorded after measurement compatibility and metadata
checks but before LKCP target/model scores are calculated. All data are MODULE_DEV.
It is not a registration of a new independent primary evaluation population.

## Frozen core, not a substitute

Use the complete saved Pilot1 STATE50 fold-0 model, selected by index rather than
LKCP performance. This includes its original RIDGE, HR, generic correction and
state correction. Verify exact replay on its original queries before transfer.
Use the same 1694 named features, DMSO normalization semantics and fixed clipping.
The core mean and preprocessing are not retrained. Original Pilot1 chemistry
overlap is reported separately: this is not an unseen-compound claim for the core.

The existing core error recipe is adapted only on declared LKCP 24-hour reference
objects: LOCAL_SCALE residual second moments with leave-chemical-group-out
weights, an amplitude-conditioned total scale, and an empirical radial law with
amplitude-conditioned retrieval. The baseline and optional arms share those
fitted quantities. Unknown structure uses the existing masked-chemistry path,
not invented fingerprints. Recorded concentration is not converted to assay µM.

## Roles set from identity, not outcomes

Partition chemical groups into five shuffled folds, seed 20260916. For each fold,
48-hour objects in its held-out groups are queries. The remaining groups at
24 hours split 60/20/20 into representation fitting, covariance fitting and
radial calibration using fixed seeded splits. Keep all positions/doses of a
chemical group together. Group aliases are unioned using exact known structures
and common compound IDs. No query future wells enter any module or calibration.

The original plan's 24-hour source / 48-hour held-condition roles remain intact.
Query chemistry groups are additionally excluded from all new 24-hour fitting
and references. There can still be compound overlap with the pre-existing
Pilot1 core. No 48-hour labels choose a model, mixing coefficient or epoch.
Use all eligible records; no outcome-based deletion. Episodes, compounds and
chemical groups are reported separately. Fixed positions and only three layout
families limit deployment claims and layout resampling.

## Arms and gates

The five arms and coefficient selection rule are exactly those in
MODULE_SWITCHES_PLAN_20260916.md: CORE, BIO, PCA_STATE, DIRECT_STATE and
CONDITIONAL_STATE. Representations train on source MODEL_FIT only, maximum60
epochs with internal group early stopping. Calibration chooses the smallest
nonzero mixing passing the paired one-SE improvement criterion, or zero.

BIO requires verified cell, time and actual dose matching. Actual dose remains
unknown; moreover source24h and query48h differ. It is therefore expected to
return exact core predictions here, irrespective of embedded annotation counts.
This is a non-applicability result, not evidence that biology is unhelpful.

The other states have only X as query input and are tested for transfer rather
than assumed to transfer. Reference retrieval changes the radial distribution,
not the mean or error directions. A positive RBF value is not proof of being in
the training domain. No enabled arm automatically becomes the primary method.

## Evaluation

Same100000 joint draws and random streams per object/arm. Score EΓ−0.2P(NULL),
apply the frozen extra-well budget B=floor(0.25×query count), then k=floor(B/2),
with ties by stable episode ID. These are developmental fold quotas, not one
physical campaign or a new formal certification. Report actual
value and NULL counts together with Γ CRPS, NULL Brier/AUC, joint NLL, energy,
five coverage levels and single/pair/average functionals. Include λ=0 and
same-budget random expectations. Off must recover the whole core exactly.

Bootstrap chemical groups and separately layout families, conditional on fixed
predictions; neither is claimed to provide a formal finite-sample certificate.
Original action cost is retained. Costs of building references and primary
certification remain separate. No old FINAL or unused fifth repeat is opened.
