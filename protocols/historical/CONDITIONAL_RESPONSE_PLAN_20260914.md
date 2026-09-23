# Conditional local responses: fixed 30-epoch development comparison

Recorded before this run, 14 September 2026. This experiment implements the
local-response interpretation of the proposed kernel; it is not nearest-neighbor
borrowing of future responses. The existing 639 opened source_5 DEV compounds,
four measured roles, five compound folds, original geometry coordinates and
ADD_TWO endpoint remain unchanged. No new independent validation is claimed.

## 1. Meaningful inputs and local response families

All input descriptors are available from the initial well and its chemical
fingerprint. Each fold uses the same 64 TRAIN anchors and descriptor scaling as
the previous replacement experiment. The 131 descriptors are:

* 64 chemical Tanimoto similarities. The structured family is T, T², T⁴: increasing
  locality emphasis, not a dose-response law or proof of shared mechanism.
  The matched generic family has Gaussian RBFs centered at 0, 0.5, 1, width 0.25.
* 64 initial-well direction cosines q. The common signed family is q, tanh(2q),
  q|q|. It preserves the distinction between concordant and opposite directions.
* Three standardized observed descriptors: initial-well log norm, log(1+norm)
  of preprocessed X, and chemical fingerprint bit density. Their common family
  is v, tanh(v), v/sqrt(1+v²). The linear term retains amplitude information.

These are declared structural priors about similarity and observed measurement
response. Current data do not establish target/pathway, dose-response or cell-
count mechanisms. Unknown mechanism annotations stay inactive. Each response
block is divided by one TRAIN-fitted uncentered RMS. No per-object normalization,
LayerNorm, softmax or coefficient L1 normalization is introduced.

## 2. Small conditional mixer, no descriptor bypass

For descriptor j and basis m, local response is

    f_j = sum_m w_jm * gate_block(j),m(X) * b_jm(d_j).

The conditional gate is 1 + 0.5 tanh(MLP(d)); MLP is 131→16 GELU→9 and produces
three coefficients for each of the three blocks. It cannot directly output the
nine-coordinate correction. A bias-free linear readout maps the 131 local
responses to nine coordinates. There is no further MLP after the local responses.
Local coefficients start at 1/3; conditioner output starts at zero (gate 1); the
readout starts at zero, exactly preserving HR. This deliberately gives staged
gradient activation, not a dead branch. Weights may be signed, so monotonicity of
individual chemical bases does not imply global monotonicity or identifiability.

The inherited mean is unchanged:

    ridge + 0.5*tanh(frozen_HR_raw + available*new_response_raw).

Thus the existing HR MLP remains a frozen global baseline, and the outer bound
still limits correction amplitude. 'No amplitude normalization' does not mean
unbounded final predictions. Chemical-unavailable rows retain the historical HR
fallback; all current 639 objects have fingerprints.

## 3. Arms and training

* A_HR: immutable historical HR predictions, no new training.
* E_STATIC_STRUCTURED: structured bases with nine learned global gate logits,
  no input-conditioned MLP. Tests fixed versus object-dependent combination.
* F_CONDITIONAL_GENERIC: conditional mixer with generic chemical bases.
* G_CONDITIONAL_STRUCTURED: identical conditional capacity, structured chemistry.

F and G are parameter matched. E has fewer parameters by design. Previous B_MLP,
C_GENERIC and D_STRUCTURED are reported as historical context from the identical
folds; their change to the new architecture is not a single-factor comparison.
In particular, the common signed morphology/scalar families also changed.

Train all 15 new fold-arm fits from their exact-HR initial state for 30 epochs,
7 batches per epoch, 210 updates per fit. Optimizer and sample order seeds match
the previous branch experiment. AdamW lr 0.0003, weight decay 0.0001, batch 64,
clip norm 5, ten-update warmup and the existing 100-epoch schedule (minimum
0.000003) are retained. Do not restart or compress the schedule to 30 epochs.
The loss remains nine-coordinate MSE + 0.1*increment MSE. No Γ-label auxiliary
loss or covariance refit is bundled into this architectural comparison.

Record validation and gradients every five epochs (training loss each epoch).
Save epoch 0,5,...,30 and validation-best with optimizer/RNG state. Main results
use actual epoch 30 for every new arm, not the test-best or validation-best arm.
Thirty epochs is a requested readout budget, not a convergence assertion.

## 4. Evaluation and attribution

Use the original frozen full RIDGE joint-error covariance and identical 2,000
Monte Carlo draws/seeds for each fold-arm. Decode valid conditional Gram matrices
in the unchanged original endpoint; do not independently combine cosines. Record
geometry MSE, Γ CRPS, NULL Brier, gain rank correlation, joint/marginal coverage,
and the original physical-well-budget policy. At 25% additional-well budget and
two wells per selected object, select 16/16/16/16/15 objects across the folds:
79/639 objects and 158 wells. Report actual selected value, population value,
FDP and false-activation risk together; do not relax purity thresholds.

Use paired within-fold object bootstrap (2,000 replicates) for G−A, G−E, G−F and
historical comparisons. These intervals condition on fitted folds and do not
include training, batch, repeated-development or Monte Carlo uncertainty.

Record gate variation across objects, local response and block contributions,
their exact additive closure in raw correction space, coefficient distributions,
branch gradient flow and correction saturation. These show whether local bases
are used, not whether biological mechanisms were discovered. A model that improves
geometry but not Γ or budget value has not demonstrated acquisition benefit.
Mean/covariance adequacy is monitored, but no post-result refit is added to this run.

## Preserved material

Original outcomes, seven-item contract, split files, old runs, FINAL400, fifth
repeat and unopened candidates remain untouched. Only new source modules, tests,
this protocol and a fresh run folder are created. Old paused training stays paused.
