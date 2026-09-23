# Full-L statistical repair: mean and conditional dispersion, separately

Date: 2026-09-13. This is a new development experiment on the already opened
639 source_5 compounds and four physical roles. It does not reinterpret an old
result as independent validation. The original 383 TRAIN, 96 validation,
96 evaluation and 64 calibration objects, measurement space, roles and utility
remain unchanged. No FINAL, fifth repeat, unexported objects, custody or source
cache is read. No neural or JEPA model is trained. No new exclusions or QC
failure labels are applied; all finite extreme observations remain outcomes.

## Sequence and baseline identity

1. Load the exact saved full-space L from moment_repair_20260913_v1, not a new
   reconstruction from reliability_spectrum. Verify all held-out means and
   analytic contrast coverage against the existing saved reference. Contrast
   residuals are formed once in physical coordinates: no duplicate slot mean.
   Neutral correction must preserve means, covariance, density and samples.
2. Audit the existing export and training scaler: extreme coordinates, source
   units, scaling denominators, feature names and available provenance. A large
   standardized value is a diagnostic flag, not proof of segmentation failure.
   Do not download images or execute unapproved QC exclusions.
3. Select two independent statistical modifications using original TRAIN only.
4. Freeze them and evaluate on all three original held-out DEV partitions.

## One internal development split

Permute the 383 original TRAIN indices with numpy default_rng(20260913).
The first floor(0.8*383)=306 form INNER_FIT; the remaining 77 are INNER_CHECK.
This does not change the original four-way partition. Fit the internal affine
transform, full L moments and PCA basis only on INNER_FIT. Save both lists.
Use the same complete L family as the stored reference: rank 200, train moment
clipping +/-8, noise shrinkage .05, full-coordinate variance floor 1e-6, seed
20260912; prediction inputs and targets are not clipped. INNER_CHECK is never
used for a feature transform, residual regression fit or moment fit.

## Three declared arms

L_REFERENCE: original saved complete L, with all 3617 coordinates and positive
residual covariance. Reuse its completed 2000-draw outputs and exact contrasts
after identity verification; do not resample an unchanged baseline.

L_MEAN_RESIDUAL: m_new(X)=m_L(X)+delta(X), with L's covariance unchanged.
The correction has rank min(64, L.rank), using the corresponding train-fitted
measurement PCA basis. It does not replace or truncate L's full mean/noise.
Input features are unclipped slot-0-centered basis scores plus log(1+mean of
affine X squared); feature standardization is fit-only. Each future role has
its own residual coefficient vector. Fit ridge regression to the actual future
measurement residuals projected into that basis, with an unpenalized intercept.
Select alpha from [1,10,100,1000] by full physical-coordinate INNER_CHECK MSE.
Retain a correction only if MSE improves by at least 1% over internal L; otherwise
the arm is explicitly the unmodified L. Refit the chosen regression on all 383
TRAIN objects using the original saved L. No separate mean/covariance repair is
silently introduced after this choice.

L_CONDITIONAL_SCALE: m_L unchanged, entire conditional covariance multiplied by
one positive scalar a. This is conditional dispersion calibration, not a claim
that a is a physical well-noise parameter or a latent c_i. Density, sampling,
all contrasts and derived Gamma use this same Gaussian. Select a from
[.75,1,1.25,1.5,2,3,4] on INNER_CHECK; no test value is fixed because a previously
viewed held-out table looked favorable. Score central intervals at 50/80/90/95%
in inner affine units using interval score (width plus outside penalties), with
equal weights on single wells, differences, pair means and triple mean groups.
Choose the lowest score; require >=1% improvement over a=1, else retain a=1.
Keep this INNER_CHECK-derived scalar for the original full-TRAIN L, without
re-estimating it from in-sample residuals or the old held-out partitions.

The mean and scale modifications are NOT combined in this run. There is no
Student-t, mixture, heterogeneous-scale network, new role-pair covariance
family or additional seed search. These are complete declared statistical
comparators, not reduced substitutes for a neural world model.

## Evaluation and reporting

For each non-neutral modification, compute exact Gaussian contrast coverage,
width, interval scores, standardized residuals and per-object traces in all
3617 coordinates. Report original-space mean MSE/R2, joint NLL, full-sample
geometry, Gamma bias/CRPS/ranking, NULL probabilities and risks, and original
5/10/25% physical-well-budget policies against fixed/random references. Preserve
2000 joint draws, 32 draw chunks, seed20260912, two-object chunks and 2000
descriptive resamples/random references from the existing evaluator. Reusing
baseline outputs or aliasing a neutral arm is explicitly recorded, not counted
as a fresh experiment. Include all three partitions and all extreme objects.

Coverage is descriptive across correlated coordinates; neither internal
selection nor historic DEV evaluation grants a certificate. Shared batches
remain fixed and are not made independent by resampling compounds. Do not
select the next model by the most favorable evaluation row or by 95% coverage
alone. Changes in interval width and realized budgeted value/risk must accompany
coverage changes. A fallback to L is an informative negative result.

Use two CPU threads without changing other jobs. Check progress after 40
minutes as requested; completion timing is not guaranteed. Stop after this
experiment and its analysis before any neural refit or repeated five-fold work.
