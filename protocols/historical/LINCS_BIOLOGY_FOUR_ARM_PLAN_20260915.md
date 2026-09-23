# LINCS Cell Painting: information versus response functions

## Question and fixed comparison

This development experiment asks whether externally annotated pharmacology adds to the initial-well morphology and chemical-structure information, and whether an explicit target/MoA relationship response helps more than a generic response to the same information. It does not presume that either will improve acquisition decisions.

| Arm | Information | Response functions |
|---|---|---|
| A_OLD_GENERIC | Initial well, chemical structure, existing reference descriptors | Existing generic local functions |
| B_OLD_STRUCTURED | Exactly A's information | Existing chemical-locality structured functions |
| C_BIO_GENERIC | A plus available external target and MoA annotations | Generic functions of the declared biological relationship descriptors |
| D_BIO_STRUCTURED | Exactly C's information | Explicit known-target overlap, known-MoA overlap, and their conjunction |

C and D keep the **old chemical functions generic**. Thus B−A tests the old structural response, C−A tests the information addition under the capacity-sharing design, and D−C tests the new response functions. This is not an orthogonal 2×2 factorial in which both old and new functions change together.

## Data selection before response inspection

Use official LINCS Cell Painting Pilot1, A549, 48 h, the repository's nominal 10 µM dose stratum, with four distinct physical wells at a matched recorded concentration. This is not an exactly equal dose across compounds: the metadata-only audit found recorded concentrations of 6.7937–11.5467 µM in this nominal stratum. Retain and report the actual per-compound concentration. Match actual compound/sample identifiers and record full chemical identity. Do not mix Pilot1 with LKCP/Batch2. Resolve any multiple eligible dose/sample entries by a declared metadata-only lexicographic rule. The four-arm chemical-kernel cohort requires a valid decision-time SMILES; report input-availability exclusions before fitting. Among that input-eligible cohort, include all compounds, not only compounds with favorable responses or mechanism annotations. Select X, Z1, Z2, V by the declared plate/well ordering, without reading response values to choose roles. Record remaining replicates but do not use them in this comparison.

Use per-well profiles, never five-repeat consensus. Verify dose units against the official IDR metadata (µM); the legacy column name alone does not establish units. Document plate and well-position reuse. If technical replicates remain at fixed positions, this is a limitation, not an intervention on batch effects.

The previous JUMP cohort, FINAL, fifth repeat, contracts, and results are outside this experiment. This public LINCS development comparison is not a new independent certification of a previously selected rule.

## Measurement and annotation processing

Use official per-plate DMSO-normalized level-4a profiles, without whole-batch whitening or treatment-based feature selection. Header inspection found 36 DNA radial-distribution measurements absent from eight plates. The recorded schema revision uses the intersection of the selected plates' measurement schemas, excluding ten object/parent/nearest-object identifier columns by name. Select the remaining Cell/Cytoplasm/Nucleus measurement features that are finite in the selected plates' DMSO controls and have control variance greater than 1e-12; do not select on treatment values or Γ. Sparse individual nonfinite treatment coordinates are filled at the DMSO-standardized center (zero), with per-well counts retained. Stop before training if a required well is absent/nonunique, has over 5% nonfinite selected coordinates, or has zero norm after filling. Do not silently discard such objects or relax this rule. Apply the same declared ±10 coordinate clipping to all four roles and all arms. Retain the original half-cosine ADD_TWO utility with cost 0.02; do not alter its sign or purity definition. This is explicitly a new dataset's preprocessing recipe, not a revision to the old JUMP endpoint.

Biological inputs come from the versioned public compound annotations. Preserve the source sample ID, structural matching level, target set, MoA set, and separate missing flags. Missing targets are not negative targets. Do not assign every compound-level MoA to every target as a verified signed edge. Do not infer Kd, occupancy, a Hill coefficient, dose-response parameters, or causal pathway direction from categorical annotations. The new functions are pharmacological relationship priors, not validated dose-response laws.

No future well, realized Γ, or outcome-derived activity label enters the biological descriptors. Target/MoA labels used as inputs will not also be called independent biological validation.

## Matched full model and training

Fit a fresh full-input ridge conditional nine-coordinate Gram model and bounded HR mean on each LINCS outer fitting partition, using the existing implementation and inner validation recipe. All four branches in that fold start from this same fitted HR. Keep its full joint residual covariance fixed across the four branches. The original-space Gram-to-utility map and joint sampling remain identical.

Use five compound-group outer folds and a separate inner validation subset. Keep the same chemical identity, including different samples of the same connectivity when present, in a single fold. Allocate 64 independent reference objects from the outer fitting pool using a fixed seeded metadata-only rule. Fit all four added branches on the remaining same objects. Reference descriptors use only initial wells and external annotations. All numerical transformations and block scales use the corresponding fitting objects only.

The fixed chemistry-eligible cohort contains 1,188 objects (1,001 target-known and 1,179 MoA-known), selected from 1,503 four-well metadata-eligible objects before response evaluation. The 315 unavailable-SMILES objects remain in the raw data but are outside this chemical-kernel comparison. Three shared-connectivity pairs remain grouped even in the internal OOF residual estimation. Reference-group exclusion concerns the added branches' supervised rows; the shared HR backbone is fitted on the full declared outer fitting pool, including reference objects.

Match the active trainable capacity, not only a padded parameter count: all arms retain the existing 131 local descriptors, three coefficients per descriptor, bounded three-block conditional gates, and bias-free nine-coordinate readout. C/D add biological basis responses at the existing chemical-reference positions and share their coefficients/gates/readout. There is no extra free biological network. Report this shared-coefficient restriction explicitly.

Each arm runs **30 epochs**, one fixed seed per outer fold, batch 64, AdamW learning rate 0.0003 with the existing warmup/cosine schedule, weight decay 0.0001, gradient clipping 5. Keep the existing geometry MSE + 0.1 incremental-mean MSE + normalized Γ-CRPS objective. All arms use the same minibatch order, Monte Carlo random streams, fitting objects, update count, and initialization. Report the fixed epoch-30 checkpoint; validation-best checkpoints are descriptive only.

## Readout

Use 10,000 joint predictive draws per object, shared random numbers across arms. Report geometry MSE, Γ mean error and Spearman correlation, Γ CRPS, NULL Brier/calibration, and the covariance diagnostics for individual geometry components, well differences and well averages. Improvements in one score do not imply improvements in another.

The principal policy comparison uses a fixed 25% **additional-well** budget: ADD_TWO costs two wells per selected object, so each outer fold selects floor(0.25×N_fold/2) objects by predicted expected Γ. Report selected actual net value, NULL count, FDP, FPR and coverage together, including HR and same-budget random baselines. Never substitute 25% of objects for this budget.

Pool exactly one out-of-fold prediction per compound. Show paired differences C−A, D−C, B−A and each arm versus HR, alongside individual-fold directions. Bootstrap intervals conditional on the fitted predictions are development uncertainty summaries, not independent CP certification; shared plates and training overlap limit their interpretation. Also report plate-layout-block sensitivity where metadata supports it.

Before training, check that target/MoA relationships actually connect fitting and held-out objects to independent references. If they do not, report a failed information-support check rather than describing disabled branches as a biological experiment. Do not change the cohort or functions after seeing the four-arm outcome.
