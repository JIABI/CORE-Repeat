# Biological-reference information increment: retrospective diagnostic

## Question and scope

Does target/MoA-indexed reference memory improve the strongest existing STATE50 prediction after generic morphology/chemistry memory receives exactly the same reference outcomes? This is an auxiliary-information diagnostic, not a new trained neural architecture or an independent certification experiment.

Use only the existing 1,188 LINCS objects, four measured roles and the five saved STATE50 models/predictions. Do not alter preprocessing, Gamma, NULL, the original contracts, previous runs or protected data. The response error measured here is the nine-dimensional legal geometry error of the current model, not full-spectrum reconstruction error.

## Honest reference records

Within each original outer test fold, split chemical-connectivity groups randomly into two halves using seed 20260916 + fold. Freeze the original STATE50. Each half is the reference set for the other, then swap. Neither reference nor query outcomes trained that base model. Reference records contain complete observed four-role geometry, its base prediction error, measured repeat contrasts, and decision-time metadata. No query future measurement is a key or a memory value.

This allocates approximately 118 fully measured references per query half. These reference measurements are additional calibration information, equally available to all memory arms; their acquisition is not free in deployment. Current scores are retrospective DEV diagnostics, not net savings after paying a new calibration campaign. Overlapping swap/fold computations create dependence; uncertainty intervals describe fixed saved predictions, not a refit or deployment guarantee.

## Arms and tuning

BASE retains STATE50. POP uses uniform reference weights. GENERIC uses positive morphology cosine and fingerprint Tanimoto, fixed top 16 with stable identity tie-breaking. BIO retains that generic path and adds target/MoA-indexed reference weights, with effective-neighbor shrinkage n_eff/(n_eff+8). No biological support recovers generic memory exactly. COUNT is a separate generic control adding first-well cell count proximity; it is not credited as a biological-kernel effect.

For mean-error borrowing, choose correction amplitude from {0,.25,.5,1}; BIO also chooses mixture strength from {0,.25,.5,.75,1}. Selection uses reference-only leave-chemical-group-out geometry MSE, with ties favoring smaller changes. Evaluate all query errors only after this choice. A fixed shuffled-annotation mean arm is a negative-control sensitivity, not a permutation significance test.

For uncertainty, borrow whole nine-dimensional residual blocks. Their weighted covariance retains cross-coordinate dependence but is a predictive error covariance, not identified biological/technical variance components. Blend with the original covariance at strengths {0,.25,.5,.75}; retain at least one quarter of the original covariance. Choose blending/biology strengths by reference-only leave-group-out multivariate Gaussian score. Report mean-only, covariance-only and combined changes separately; covariance-only preserves the original mean. A Gaussian diagnostic here deliberately does not implement a general non-Gaussian memory model.

## Outcomes

Implementation clarification before completed scoring: auxiliary contrast tuning uses raw squared error and group-excluded reference means, not centering fitted to the held-out reference itself. Unsupported biological cases recover the tuned generic path during both reference tuning and query prediction. Uncertainty moments are taken around the mean actually used: weighted centered covariance plus the outer product of the remaining weighted residual bias. Combined-arm uncertainty is tuned after its selected leave-group-out mean correction.

1. Remaining geometry-error MSE, direction alignment and correction size; primary biological contrast is BIO minus GENERIC, not BIO minus BASE alone.
2. Independent descriptive prediction tasks: future-pair log squared distances and sign-invariant directional second moments in an eight-dimensional PCA coordinate system fitted only to original FIT first wells. Compare POP, GENERIC, BIO and COUNT. These are noisy observed contrasts, not pure technical-noise labels or signed future-noise predictions.
3. Draw 10,000 legal joint geometries per object from each declared Gaussian mean/covariance combination. Compute the unchanged ADD_TWO Gamma per draw, then Gamma fair CRPS, NULL Brier score, intervals and predicted value. Verify the forward geometry identity against existing code. Do not average donor Gamma to substitute for this calculation.
4. Keep the existing 146-object total selection budget, but allocate it deterministically across query halves before reading scores. Rank only within each query half, including for BASE: pooling swapped halves before ranking could allow a query's outcome to affect competitor scores. Report selected mean realized Gamma, total net value per eligible object and NULL counts. Reference acquisition costs are not included in this diagnostic policy budget.

Report chemistry-group paired intervals and layout-cluster sensitivity for objectwise scores. Neither includes retraining uncertainty. Do not remove hard objects, change endpoint normalization, or interpret absence of a detectable gain as proof that all biological knowledge is useless.

## Noise interpretation

Keep control-based correction distinct from erasing biological variation. Preserve first-well count and response magnitude as candidate covariates. Residual error can contain omitted mean structure, finite-sample estimation error, biological heterogeneity and technical noise. Fixed positions and shared plates prevent a causal separation here. The current well profiles are medians, so a strict variance proportional to 1/cell-count law is not assumed.
