# Joint-tail calibration with a frozen STATE50 mean

## Question

The local reference-scale model improves proper distribution scores but covers
only 88.1% of observed nine-dimensional geometry in its nominal 95% Gaussian
ellipsoid. Can a separate calibration reference set repair this discrepancy,
and does changing the probability distribution help or hurt the acquisition task?

The experiment uses the opened LINCS Cell Painting 1,188 objects only. It keeps
the original endpoint, four physical roles, STATE50 checkpoints, preprocessing,
outer folds and query/reference cells. It does not train a neural model, access
FINAL or a fifth replicate, remove difficult objects, or activate JEPA/biology.

## Reference fitting and calibration

Within each of the ten existing donor/query cells, shuffle the sorted unique
donor chemical-connectivity groups with seed 20260917 + 100*fold + half. Hold
out ceil(number_of_donor_groups/3) groups for calibration; use the rest to fit
the reference-error model. Do not use outcomes to allocate groups.

Fit the complete LOCAL_SCALE procedure on the fitting reference pool: the
existing direction/chemistry/amplitude retrieval, top 16 eligible references,
and beta in {0, .25, .5, .75} selected by reference leave-chemical-group-out
Gaussian likelihood. Estimate the amplitude bandwidth from the original base
model's training X only. The query and calibration covariances use this fitted
reference pool, not one another's outcomes.

The mean and original covariance come from the same frozen outer-fold model.
Both calibration and query objects were excluded from that model's fitting.
Use one deterministic, lexicographically first ID per calibration chemical group
for the scalar calibration scores; retain all query objects for evaluation.

For calibration residual r and predicted covariance C, compute a=r' C^-1 r.
For m independent calibration units, the finite rank is ceil((m+1)*.95).
The empirical squared-radius threshold q is that ordered score; if the rank
exceeds m, the region is unbounded rather than silently using the maximum.
The primary experiment reports empirical coverage. Shared plates/layouts and
repeated use of DEV prevent treating this as a new distribution-free certificate.

## Four matched arms

1. **LOCAL_FIT:** Gaussian conditional error from fitting references only;
   nominal chi-square(9) ellipsoid and original distribution-derived intervals.
2. **REGION_ONLY:** identical mean, covariance, density, samples, Gamma,
   NULL probabilities and selection; replace only the joint region radius by q.
3. **GAUSSIAN_Q95:** multiply LOCAL_FIT covariance by q/chi-square(9,.95).
   This has the same joint ellipsoid as REGION_ONLY, but changes the full law.
   Matching one quantile does not calibrate its entire distribution.
4. **GAUSSIAN_MLE:** multiply LOCAL_FIT covariance by mean(a)/9, using the
   same calibration representatives. This is the fixed-mean scalar Gaussian
   likelihood fit; it contrasts center/likelihood fitting with tail fitting.

No scalar is clipped to be at least one, and no query metric selects an arm.
Nonfinite or nonpositive full-law scaling stops the run rather than creating
an artificial covariance. Historical BASE and all-donor LOCAL_SCALE results
are contextual references, not matched calibration controls.

## Readouts

Evaluate every object once with 10,000 common Gaussian draws per distribution.
Keep each cell's existing selection budget (146 total objects, 292 added query
wells) and rank within that cell. Reference acquisition costs remain excluded.

Report mean MSE (must be identical), joint and coordinate coverage, squared
Mahalanobis quantiles, ellipsoid radius/volume changes, NLL, energy score,
single/difference/average observables, Gamma CRPS, NULL Brier, Gamma ranking,
selected actual value and NULL count. REGION_ONLY must match LOCAL_FIT
distribution and selection exactly; its region must match GAUSSIAN_Q95.

Use paired fixed-prediction chemical-group and layout-block bootstrap intervals
for score/value differences. These intervals do not incorporate refitting,
calibration-set uncertainty or repeated model development. Show all ten scalar
estimates and calibration sizes to expose calibration instability.

Separately describe existing residual tails by observed-X amplitude quartiles,
fold, layout and whitened directions. These are diagnosis, not exclusion rules
or new subgroup-specific calibrators. Do not infer a biological variance
decomposition from predictive residuals alone.

## Interpretation decided before execution

- Better joint-region coverage alone supports uncertainty-set calibration,
  not more accurate NULL probabilities or more valuable selection.
- Worse CRPS/Brier after whole-law scaling means a tail threshold cannot be
  used as a global distribution repair; retain the distinction in the report.
- If residual direction/shape remains misspecified, propose a separate
  reference-block/radial-shape experiment. Do not tune it on this run's queries.
- No change to the original acceptance contract follows from this diagnostic.

Calibration background: Angelopoulos and Bates, *A Gentle Introduction to
Conformal Prediction and Distribution-Free Uncertainty Quantification*,
https://arxiv.org/abs/2107.07511 . The exchangeability requirements are not
established for this shared-layout retrospective dataset.
