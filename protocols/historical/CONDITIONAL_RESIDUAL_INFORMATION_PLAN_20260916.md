# Information beyond amplitude in frozen CORE prediction errors

Recorded 2026-09-16 before this experiment. Population: the 1,188 already opened
LINCS Cell Painting objects. Existing five chemical-group outer folds and ten
reference/calibration/query cells remain unchanged. No protected JUMP objects,
fifth repeats, or new evaluation population are opened.

## Questions and fixed quantities

1. Do measured, decision-time descriptors explain predictive dispersion beyond
   first-well amplitude and assay conditions?
2. Does a four-dimensional learned first-well state add to a strong nonlinear
   descriptor model?
3. Do distribution changes improve Gamma, NULL probabilities, or the original
   fixed-budget decisions?

Outer STATE50 means, nine-dimensional legal geometry, ADD_TWO utility and .02
action cost, score E[Gamma]-.2 P(NULL), stable-ID ties and ten budgets (146 actions)
are fixed. Reference costs remain a separate deployment cost, not included here.
This is development evidence, not a new independent certificate.

## Honest residual training

Within each original MODEL_FIT pool, perform three chemical-group inner folds.
Refit the complete existing mean recipe (selected ridge, HR, old generic branch
30 epochs, STATE50 branch 50 epochs), with its own fitting, validation and 64
reference roles. The held-out inner identities enter none of these roles.
Return predictions to native geometry before combining folds. Do not replace
STATE50 with a linear surrogate, use in-sample residuals, or pool historical
outer OOF predictions whose models saw the present outer queries.

These auxiliary refits produce training labels only. Existing deployed outer
checkpoints and query means are unchanged. The smaller inner training size is
reported; independent DIST_CAL still estimates the final radial distribution.

## Inputs

All fitted descriptor transformations use only original MODEL_FIT objects.
Inputs are first-well amplitude and legal assay conditions; segmented cell
count; summaries of within-object intensity/texture descriptors; training-set
distances; energy allocation in training-derived reliability directions; and
available normalized same-X-plate DMSO summaries. Raw plate identity is excluded.

MADIntensity, RadialCV and texture variance are not called between-cell
heterogeneity. Original DMSO variance, single-cell dispersion, focus and saturation
are marked unavailable if not recorded. Normalized controls describe remaining
shape/tail structure, not recovered raw technical variance. Potency annotations
are audited for availability only; no EC50 activation window is fitted this round.

## Common distribution extension

Whiten each honest training error by its inner predictive covariance. The
Jacobian of the three future-well difference energies at the predicted mean
defines a rank-three orthogonal projector P; I-P has rank six. Targets are
the two squared projection norms, with degrees of freedom 3 and 6. They are
prediction-error summaries, not identified physical independent/shared noise.

Use positive-scale Gamma-deviance gradient boosting separately for the two
energy-per-degree targets: 150 iterations, learning rate .05, at most 7 leaves,
minimum 20 samples per leaf, L2 regularization 10, no outcome-selected stopping.
An amplitude/condition predictor is the common denominator. An added-information
model predicts the same two targets. Its variance ratios to the denominator are
bounded smoothly using exp(log(4)*tanh(log(ratio)/log(4))). This is the declared
extension parameterization, not clipping observations or sampled geometry.

Apply these two ratios to the existing CORE scatter in the corresponding
query-specific geometric subspaces. Mean remains exactly fixed. On the existing
DIST_CAL identities, recompute whitened radii under that arm, fit the same
empirical log-radius law (same triangular bandwidth rule and 10% Gaussian guard),
and retain the original amplitude-reference weights. All arms therefore share
the predictive family, reference resources, radial calibration and full-vector
sampling interface. The family does not model arbitrary radius-direction
dependence; angular diagnostics delimit what this comparison can establish.

Disabling the extension returns original mean, scatter, empirical law and
predictions exactly. CORE is replayed with the existing scoring implementation.

## Prespecified comparisons

- CORE: existing amplitude-conditioned model, unchanged.
- DESCRIPTORS: full legal descriptor boosting, relative to amplitude/conditions.
- DESCRIPTORS_LATENT: the identical boosting predictor with an additional learned
  four-dimensional state; all DESCRIPTORS inputs remain available.
- Five block removals from DESCRIPTORS: cell count, texture/intensity summaries,
  train-distance, reliability energy, and normalized control/context summaries.

The latent encoder takes training-fitted X direction PCA32 plus explicit amplitude,
uses a 32-unit GELU hidden layer and a four-dimensional bottleneck. A two-scale
head also sees the descriptors. Train on the honest residual projection energies
with the Gaussian projection negative log likelihood, AdamW learning rate .0003
with five-epoch warmup and cosine decay, weight decay .001, batch 64, 60 epochs.
The baseline amplitude prediction is an offset, and scale corrections are bounded.
The encoder is not a moving-teacher JEPA and is not trained to reconstruct future
profile means. No latent dimension is required to have unit variance. Fixed
training length avoids checkpoint selection on a residual validation set coupled
through upstream cross-fitting. Save losses every ten epochs and the final model.

## Evaluation

Use 100,000 complete nine-dimensional draws per object with the existing common
random streams and original Gamma map. Primary information comparisons are
DESCRIPTORS minus CORE and DESCRIPTORS_LATENT minus DESCRIPTORS. Report joint NLL
and energy, five coverage levels (50/80/90/95/99), widths where available,
single-well/pair/mean functional scores, Gamma CRPS, NULL Brier/AUC and unchanged
full-cohort budget value/NULL count. Angular-energy and residual-centering checks
separate scale information from unmodeled bias or directional structure.

Report paired chemical-group uncertainty and layout sensitivity, conditional on
fitted predictions. No query-based winner selection, post-hoc removal of difficult
objects, or rewriting of old results. A distribution improvement without decision
improvement is reported as such. An unresolved contrast does not establish that
first-well information or biological representation is universally useless.

The separate matched-random biological-reference experiment has its own protocol
and does not select any of these predictor configurations.
