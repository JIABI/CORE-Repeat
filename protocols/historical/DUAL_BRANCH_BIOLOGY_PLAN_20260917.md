# Frozen CORE with an optional dual-branch biological adapter

This is a development comparison on the 1,188 already-opened LINCS objects. It does not change the main method, original measurement endpoint, action cost, cohort budgets, or protected evaluation data. JUMP FINAL, fifth repeats and unopened people/compounds are out of scope.

## Questions and fixed comparisons

1. Do target/MoA references share the remaining **mean-error direction** of CORE? The separate direction experiment uses native nine-dimensional geometry, honest held-out reference errors, matched random references, and calibration-selected mean shrinkage. Residual cosine alone is not success; actual mean squared error must improve. No resulting mean correction is imported into the distribution experiment.
2. Does an optional two-branch adapter improve CORE's joint error distribution? CORE is external and frozen. Compare CORE, CORE + generic descriptor GELU, and that identical frozen GELU branch plus either a generic reference basis or a structured reference basis. Both right branches receive exactly the same target/MoA reference summaries and have the same trainable parameter count.
3. More elaborate gates or JEPA are conditional next steps, not extra arms of this round. They are not launched if the tested additions fail to demonstrate a reproducible development increment.

## Data and information boundaries

Reuse the five chemical-group folds and ten query/calibration cells. All preprocessing is fit on the original MODEL_FIT objects. Query outcomes never choose descriptors, weights, checkpoint, shrinkage or policy. The task is development validation, not an independent certificate. Chemical-group and layout bootstrap intervals are descriptive conditional on these shared folds.

For adapter training, reuse the 15 saved full inner STATE50 models. Within each model's held-out chemical groups, split query/reference roles and build the same LOCAL_SCALE → AMPLITUDE_TOTAL → empirical amplitude-conditioned radial CORE distribution. The construction is repeated with the roles swapped. A query's reference residuals come only from the same inner held-out group and its disjoint DIST_FIT pool. Do not retrieve another row's globally assembled calibrated covariance, since its construction could have used the current query. This avoids both indirect outcome leakage and the prior RIDGE-to-CORE scatter mismatch.

For outer query/calibration objects, biological records come only from MODEL_FIT honest raw residuals, never from query/calibration outcomes. Exact recorded cell line, nominal exposure time and dose must match, and a different chemical group is required. The existing 40 disjoint calibration representatives still define CORE's radial law. A lack of eligible biological support disables the entire optional adapter exactly.

Calibration inputs use their **current cell's** DIST_FIT scatter and leave-chemical-group-out empirical-radius multiplier. An object's globally cached covariance from the opposite query/calibration role must not be used to choose current-cell shrinkage, since that cache can include the present query population's errors. Run v1 was stopped when this integration issue was found; its selection/output files are excluded. Run v2 reuses only unaffected MODEL_FIT branch weights and nested distributions and rebuilds all calibration inputs and evaluations.

The learned reference bank has smaller same-inner-fold support during training than at outer inference; report this difference. Inner mean models also have smaller training populations than the outer CORE. Neither difference is hidden by pooling outcomes across folds.

## Inputs and architecture

Left branch: the existing 77 legal descriptors (amplitude, context, measured cell count, texture summaries, distance, reliability and normalized control summaries) → 16-unit GELU → two bounded log-scatter corrections.

Right branch: target and MoA separately supply availability, count, absolute relation mass, effective neighbour count, mean relation strength, two donor projected-error energy summaries, their uncertainty summaries, amplitude/direction mismatch and confidence. All donor errors are expressed in the **query's** native geometry/whitened frame. Each channel has 12 explicit fields. No potency or single-cell heterogeneity is invented.

Generic and structured right branches use four basis functions per input, independent local coefficients and the same two-output readout. Structured bases encode finite-reference support, saturation and error-scale shrinkage hypotheses. These are statistical modelling choices informed by relation support, not established biological laws. Readouts start at zero; hidden/local coefficients do not all start at zero. Each branch is bounded to ±log(2), so their sum permits variance multipliers between 1/4 and 4.

The two outputs act on the existing rank-three pair-sensitive and rank-six remaining geometric subspaces. These are **not** identified physical independent/shared noise components. The fixed empirical radial law and mean remain unchanged. A scatter change is applied jointly to all nine coordinates, followed by legal Gram reconstruction and the original Γ calculation.

Fit the left once per outer fold, then freeze and copy it identically into both right-branch arms. Fit right branches independently using the same data, optimizer, 60-epoch budget and initial seed. Use projection Gaussian moment loss relative to the full CORE covariance plus bounded-increment regularisation; this is not Γ training. Use AdamW, warmup/cosine decay, saved learning rate/loss/gradient histories. Training ends at the declared 60 epochs, not at the best query result.

## Strength and calibration

Primary mechanistic comparison uses fixed full-strength trained branches. This tests whether the proposed functions contain an increment before attenuation can conceal it. Secondary deployment-style versions select one scalar for the **whole** optional adapter from {0, 0.25, 0.5, 1}, separately for each arm, using only calibration errors. Refit no branch on query data.

For every calibration representative, construct the baseline radial law from the other chemical groups and evaluate the candidate scatter under that same law. Use paired group losses, include zero, admit a nonzero strength only if its improvement over zero exceeds one standard error, and prefer the smallest strength within one standard error of the best. This is development model selection, not a conformal or risk guarantee. After selection, retain the original full-calibration CORE radial law; do not recalibrate unsupported objects or globally change the CDF.

## Evaluation

Use 100,000 joint samples per object and the existing common random streams. Unchanged objects reuse the exact CORE distribution and saved scores, after checking the forward distribution is identical. Fixed policy remains score E[Γ] − 0.2 P(NULL), original cell budgets and ID tie-breaking.

Primary distribution readout is the pre-outcome supported subset: Γ CRPS, joint NLL and energy; NULL Brier and five-level region coverage are co-reported. Full-cohort metrics, supported and full mean errors, selected NULL count, realised net value, membership changes and all five folds are reported. A small supported-subset improvement does not establish a cohort allocation gain. Check the single-hole, pair-difference and average observable families too.

Report four distinct possible outcomes: (a) generic carrier fails CORE: no adoption; (b) right branch improves the carrier but remains below CORE: incremental effect, not a useful final model; (c) both functions improve similarly: evidence for information, not structured-function superiority; (d) structured exceeds same-input generic and CORE across held-out scores without calibration/decision harm: a candidate for new-data replication. An interval crossing zero is unresolved, not proof of equivalence. The main CORE stays frozen regardless of this repeatedly inspected development round.

## Engineering checks

Test whole-module off = CORE exactly; unsupported = CORE; right off = identical frozen CORE+left; legal positive-definite scatter; first-well-only query features; reference-group isolation; no target-dependent sample generation; same-input/same-parameter right comparison; live gradients; save/reload equality. These are implementation checks, not scientific evidence.
