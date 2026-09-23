# LINCS independent kernel: fixed input-scale calibration

## Question and sole change

Does a TRAIN-fitted, zero-preserving scale at the kernel readout improve the optimization and held-out usefulness of the independent old-information or biological correction?

Keep the neighbor L1 weights, response functions, separate Target/MoA channels, support shrinkage, learned gates, zero-initialized readouts, original A model and joint covariance unchanged. Add no trainable parameters. This is an optimization/parameterization comparison, not additional biological information.

For each channel c, let B_ijm = (w_ij / sum_j w_ij) phi_m(w_ij), zero when there is no support. On the original commonbranchFIT inputs only, compute the initial local vector h0_ij = sum_m B_ijm / 3. Define s_c = sqrt(mean over supported FIT objects of sum_j h0_ij^2). Use the fixed gain a_c = min(1/s_c, 32), equivalent to a scale floor of 1/32. A channel with no supported FIT query uses gain 1 and is recorded as unsupported. The maximum gain 32 is set before fitting or inspecting new outcomes as a numerical amplification limit, not optimized on Gamma. Report if it binds.

Multiply B by a_c before the existing learned local coefficients and readout. Do not center; do not normalize individual queries or individual reference columns. Preserve relative magnitudes across objects within a channel. Gain estimation receives no target, future well, realized Gamma, validation row or outer-test row. Gains are fixed throughout optimization and inference and saved in checkpoints.

## Arms and matched comparison

Reuse the completed run `runs/lincs_independent_biology_20260915_v1`:

- A_FROZEN: complete original A, with its old-information path.
- A_PLUS_OLD: original independent old-information correction, saved epoch30.
- A_PLUS_BIO: original independent biological correction, saved epoch30.

Train from the same zero-readout initialization, not from epoch30:

- A_PLUS_OLD_SCALED: A_PLUS_OLD with fixed channel scale calibration.
- A_PLUS_BIO_SCALED: A_PLUS_BIO with fixed channel scale calibration.

The old-information and biological arms use identical active parameter counts, original fold seeds, batch orders, objective Monte Carlo streams and monitoring streams. Compare each scaled arm to its corresponding unscaled arm, and compare the two scaled arms to each other. Include A as the frozen reference. Scale calibration does not equate evidence reliability between channels; the unchanged support gate still controls trust.

## Data, training and evaluation

Use the same 1,188 opened LINCS Cell Painting objects, original five chemistry-group folds, commonbranchFIT, inner-validation rows, 64 references, preprocessing and role assignments. No added objects, references, annotations, target changes or split changes. References remain excluded from new-branch supervised FIT as before; this does not retroactively change how the inherited HR was trained.

Each new fit runs exactly 30 epochs, batch64, AdamW learning rate0.0003, minimum0.000003, weight decay0.0001, warmup10 steps, clip5. The existing cosine horizon is 100 epochs and remains unchanged so the sole intervention is scale. Validation is recorded every5 epochs. Fixed epoch30 is primary; validation-best checkpoints are descriptive, not substitutes selected with outer-fold results. Two arms times five folds = ten new fits.

Objective remains geometry MSE + normalized joint Gamma fair CRPS +0.1 increment MSE. Training uses64 Monte Carlo pairs, monitoring128 fixed pairs. The conditional covariance and target geometry scale are unchanged. Final evaluation uses the original10,000 joint draws and shared random-number convention.

Report geometry MSE, Gamma CRPS, NULL Brier, gain rank correlation, and the original matched extra-well-budget selection. The primary budget is25% additional physical wells per fold; ADD_TWO uses two wells per selected object. Do not replace it with25% object coverage. Report selected actual net gain, NULL count/FDP and false-activation rate together.

Chemistry-group paired bootstrap and layout-group sensitivity use the original method and2,000 resamples. These intervals describe fixed OOF predictions and do not include refitting or repeated development selection. Keep the same supported/unsupported biological object subsets across arms. Record per-fold scale/support/gain, before/after local RMS, actual mean increment, gate state, fit/validation trajectories and saturation.

## Interpretation

- Larger activation without improved validation or decision metrics demonstrates scale change, not useful biological information.
- Improvement for old information alone indicates a general numerical bottleneck rather than biological specificity.
- Geometry improvement without Gamma or policy improvement does not count as successful acquisition improvement.
- No stable difference remains a valid outcome; do not choose new gain limits, favorable folds, checkpoints or thresholds after reading it.

This is a matched development experiment using an already studied cohort. Fixed well positions and shared plates/layout limit cross-batch claims. Original results, endpoint, cost, seven-part contract, protected FINAL and fifth repeats stay unchanged. No formal certificate is issued by this comparison.

## Execution artifacts

New source/run module: `opal2.independent_biology_scale_experiment`.
New run directory: `runs/lincs_independent_biology_scale_20260915_v1`.
Each fit saves scale metadata, checkpoints and numerical diagnostics; the run saves an executable source snapshot, all five-arm comparisons and a readable results report. Existing run artifacts are read-only. Old checkpoints are not redundantly copied into the new run; source paths identify them.
