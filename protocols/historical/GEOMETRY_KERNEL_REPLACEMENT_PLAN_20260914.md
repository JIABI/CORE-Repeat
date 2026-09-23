# Single-path kernel replacement: ten-epoch development comparison

## Question and scope

Does replacing the added dense descriptor path with an explicit response-basis layer improve the frozen HR geometry mean? The historical HR remains intact. This is a new, fully implemented branch with a user-requested ten-epoch readout, not a convergence claim or a publication comparison.

Use only the 639 already-opened source_5 DEV compounds and the existing X/Z1/Z2/V roles. Reuse repeat 0, all five folds, HR_VALID_S0 and its fixed RIDGE_VALID full nine-coordinate OOF error covariance. FINAL, the fifth repeat, unopened candidates, historical results and original seven-item contract are unchanged.

## Four arms

| Arm | New branch |
|---|---|
| A_HR | None; restored historical HR and saved predictions |
| B_MLP | Standardized descriptors → Linear 32/GELU → Linear 32/GELU → zero-initialized Linear 9 |
| C_GENERIC | Explicit generic chemical and common morphology/scalar bases → KAN 24/tanh → KAN 32/tanh → zero-initialized Linear 9 |
| D_STRUCTURED | Same as C, replacing only chemical response functions with T, T², T⁴ |

C and D have identical dimensions and trainable parameter counts. B is the original descriptor-only dense comparator, with fewer parameters; comparisons with B are architecture comparisons, not matched-capacity claims. No kernel arm has an additional raw-descriptor bypass. The shared KAN mixer consumes basis features only; its flexible spline/base functions are not themselves biological laws.

## Inputs and response bases

Reuse the TRAIN-only anchor identities, chemical fingerprints, morphology landmark directions and descriptor standardization. Initial-well morphology retains both direction and magnitude. Chemical similarity is Tanimoto to the same training anchors in both kernel arms.

- Structured chemistry: T, T², T⁴ for each anchor.
- Generic chemistry: exp(-0.5 ((T-c)/0.25)²), with c in {0, 0.5, 1}, for each identical anchor.
- Common morphology: RBF responses to each raw initial-well landmark cosine with centers {-1, 0, 1} and width 0.5.
- Common scalars: v, tanh(v), v²/(1+v²), for each of the three standardized scalar descriptors (observed log-norm coordinate, log1p initial-input norm, fingerprint density).

For each mode, divide each entire chemical/morphology/scalar block by its uncentered RMS estimated exclusively from available TRAIN rows, with a numerical floor of 1e-6. The common blocks and scales are identical across C/D. No per-object normalization discards amplitude. Missing chemistry uses the inherited availability fallback to HR; no unsupported target/pathway annotation is introduced. The response layer is not asserted to be a positive-definite covariance kernel.

## Mean and training

mu = ridge(X) + inherited_bound * tanh(frozen_HR_raw(X) + available * new_raw).

The inherited bound is 0.5. Only the new branch trains. The loss is nine-coordinate MSE plus 0.1 times the squared prediction increment relative to frozen HR. The output layer starts at zero; all four arms initially equal HR. Remove the old coefficient L1 normalization and the parallel dense/basis addition.

Same fold seeds and sample order for new branches. AdamW, lr 0.0003, weight decay 0.0001, batch 64, clip norm 5. Warmup 10 optimizer steps then cosine schedule with the original 100-epoch horizon and minimum lr 0.000003. Run exactly ten epochs; retain checkpoints 0/5/10 and validation-best. Main comparisons use actual epoch 10, not the test-best or validation-best checkpoint. Report parameter counts and all fold trajectories.

## Evaluation and interpretation

All arms use the unchanged full joint covariance and the same 2,000-draw seeds. Decode the legal Gram distribution and evaluate the original utility, NULL probability, CRPS, Brier and rank association. Same 25% physical-well budget: within each test fold select floor(floor(0.25*n)/2) compounds, totaling 79 compounds and 158 wells; retain risk and actual net value together.

Report all six pairwise contrasts, with 2,000 paired compound bootstrap samples within folds and ratio denominators recomputed. These conditional DEV intervals do not cover shared-batch dependence, training variation, historical model search or Monte Carlo error. No new independent certification is claimed.

The main mechanistic contrast is D versus C. C/D versus B tests the whole basis-layer replacement, with differing parameter counts. Any nonzero kernel contribution alone is not evidence of better prediction. Chemical, morphology, scalar and constant-output behavior will be inspected without selecting test-favorable configurations. No continuation beyond ten epochs or additional grid search is automatic.
