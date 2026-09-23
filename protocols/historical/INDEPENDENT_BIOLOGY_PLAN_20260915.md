# LINCS independent biological correction: 30-epoch development comparison

## Question and arms

Does separately parameterized, support-aware target/MoA information improve the original acquisition decisions beyond a frozen old-information model and an equally sized old-information correction?

1. A_FROZEN: the complete saved A_OLD_GENERIC epoch30 checkpoint from each original LINCS fold.
2. A_PLUS_OLD: A plus a newly trained independent chemical/morphology-relation correction.
3. A_PLUS_BIO: A plus a newly trained independent target/MoA-relation correction.

The two new corrections have identical active capacity, initialization, optimizer settings, training identities, and 30-epoch budget. The complete A, its reference bank, preprocessing and original joint error law remain frozen. HR and RIDGE remain historical reference models, not newly selected backbones.

## Data and information

Use only the previously opened 1,188 LINCS Pilot1 Cell Painting compounds in `lincs_biology_four_arm_20260915_v1`: all five existing chemical-connectivity folds, inner-validation partitions, four roles and 64-reference selections remain unchanged. No new cohort or compound exclusion is introduced. Existing reference-group exclusions apply to new-branch supervision; the frozen HR component was originally fitted on the original FIT pool, including reference groups. This is not a fully reference-label-disjoint pipeline.

Biology uses only curated target and MoA annotations already present at decision time. Unknown annotations differ from known annotations with no matching references. The target and MoA channels do not require one another or chemical availability. No future query measurement, realized Gamma, or target-derived noise is a decision input. Reference profiles supply initial-well directions and relations, not future-response or noise labels.

## Architecture

Let frozen A return ridge mean r, inherited coordinate bound b and total raw correction z_A. New output is `r + b*tanh(z_A + delta_1 + delta_2)`. Each independent channel has its own local coefficients, bias-free nine-coordinate output and support-conditioned gate. Each channel contributes at most 0.5 in raw coordinates, preserving the inherited total bound around RIDGE. Disabling the addition, including after training, returns exactly A. No match returns A for that object.

The two old-information control channels are chemical Tanimoto and nonnegative initial-well morphology similarity. The two biological channels are target-set and MoA-set cosine similarities; there is no target-times-MoA channel in this experiment. Both modes use the same bounded zero-at-zero response family: w, w squared, and a zero-anchored RBF rescaled to equal one at w=1. These are relation-response functions, not measured occupancy or pharmacological laws.

Relation aggregation uses query-local bounded weights rather than a sparse global biological RMS. The support gate uses six descriptors: annotation/measurement known status, positive-match status, mean relation weight, maximum weight, effective neighbor count divided by reference count, and positive reference count divided by reference count. Effective count is `(sum w)^2 / sum(w^2)`, a weight-concentration descriptor rather than an independent sample count. The multiplicative support factor is `has_match * n_eff/(n_eff+2) * sigmoid(gate_MLP)`. Each gate is independent; chemical and biological weights do not compete through a cross-modality softmax.

Output matrices initialize at zero, while local coefficients and supported gates are nonzero. Penalize only the new mean increment relative to frozen A, not relative to HR.

## Training and readout

Reuse the existing weighted objective: standardized nine-coordinate geometry MSE + original ADD_TWO Gamma CRPS divided by FIT Gamma standard deviation + 0.1 times the mean squared new increment. The joint covariance and original utility cost 0.02 stay unchanged. Use AdamW, learning rate 0.0003, floor 0.000003, original 100-epoch cosine schedule horizon, 10-step warmup, batch 64, weight decay 0.0001, gradient clipping 5, and the same per-fold random-stream conventions. Run exactly 30 epochs, recording validation every five. Epoch30 is the primary comparison; validation-best including epoch0 is descriptive and cannot be substituted after viewing test results.

Evaluate all held-out objects with 10,000 complete joint geometry draws and the original evaluator. Report geometry MSE; original Gamma fair CRPS, bias and rank correlation; NULL Brier/calibration; marginal and joint coverage; and actual selected value and NULL count at matching 5%, 10% and 25% extra-physical-well budgets. ADD_TWO costs two wells: 25% well budget selects approximately 12.5% of compounds, not 25%. The main 25% comparison retains the original fold-specific rounding and tie handling.

Report paired chemical-connectivity-group bootstrap intervals and separate layout-block sensitivity, conditional on saved fold-out predictions. Report the supported subset without dropping unsupported or difficult compounds. Preserve all fold directions, including deterioration.

## Checks and interpretation

Before assay training, test serialization, zero-initialization identity, restored baseline after optimization, frozen base/buffer identity, support masking, nonzero learning gradients and the matched parameter count. These tests are implementation checks, not experimental evidence.

The primary information comparison is BIO versus OLD; BIO versus frozen A alone does not isolate additional capacity. Better geometry without better Gamma/risk/selected value does not establish better acquisition. Unchanged or worse results are retained.

This first comparison isolates independent mean pathways. Model-specific full-pipeline out-of-fold covariance refitting follows only after inspecting mean behavior; no covariance is fitted to outer test residuals here. The current shared-plate/fixed-position limitations remain, and these reused development folds are not a new independent validation or a formal certificate. Original endpoints, seven contract criteria, prior results, protected JUMP FINAL and fifth repeats are unchanged.
