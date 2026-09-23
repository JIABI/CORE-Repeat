# CORE null-oracle and direct-risk scale calibration

Frozen before the new runs on 17 September 2026. This is development work on
the already opened LINCS 1,188 objects. CORE, the original endpoint, biological
annotations, historical results and protected evaluation partitions are unchanged.
No new world-model or JEPA training is part of this round.

## Questions

1. How much hindsight CRPS improvement does choosing a scale after seeing one
   outcome produce under CORE's own predictive distribution?
2. Can a one- or two-parameter scale function, fitted to average CRPS directly
   without using the outer evaluation outcomes, improve on CORE?

These are different questions. The first does not estimate a learnable ceiling;
subtracting its answer from observed hindsight improvement will not be labelled
an information bound. The second is an ordinary out-of-fold development test.

## 1. Self-simulation reference

Use the original ten outer cells and frozen nine-dimensional CORE geometry and
empirical radial laws. For each query generate 20 pseudo-future geometries and
20 independent second realizations, then compute the original Gamma. Do not
replace the geometry or Gamma by a Gaussian scalar surrogate.

Search the same nine-by-nine grid of two log-variance increments in
[-log(4), log(4)] using 4,096 draws. H1 uses equal increments; H2 can use different
increments. The support mask is the previous 1,047-object mask; the remaining
141 objects receive exactly zero increment. Include the actual recorded outcomes
as a separate hindsight reference. This is not a deployable arm.

Evaluate chosen candidates with an independent 100,000-draw stream. Report both
their scores against the realization used for selection and against the independent
second realization. Separate truth generation, search and final integration seeds.
Candidate draws may be reused across pseudo-outcomes; report that integration
noise is shared and is not captured by variation across the 20 realizations.
Use 20 integration blocks to report paired numerical uncertainty. Report Gamma
and joint-region coverage alongside scores; do not promote a hindsight-selected
policy as an executable policy.

## 2. Direct-risk, low-parameter calibration

Keep the five chemistry-group outer folds and the existing full CORE prediction
pipeline. Within each outer MODEL_FIT pool reuse the saved nested CORE predictions
and the six already isolated conditional-distribution cells, approximately 760
objects per outer fold. Their mean, scatter and radial laws exclude their own
outcomes. No outer QUERY or outer CAL outcome enters this new fit. The old 400 CAL
records are not pooled across outer folds.

Compare four predeclared arms, all sharing CORE's mean and empirical radial law:

- CORE: zero increment.
- DIRECT_GLOBAL: one common scalar log-variance increment.
- DIRECT_AMPLITUDE: an intercept and slope on the centered empirical rank of
  decision-time log amplitude.
- DIRECT_DESCRIPTORS: an intercept and slope on the centered empirical rank of
  the existing descriptor-family prediction of rank-six error energy.

All scales multiply both geometric blocks equally. This is generic calibration,
not a test of biological relations. It therefore applies to all 1,188 objects;
also report the prior 1,047-object support subset for comparability.

The descriptor-energy predictor for inner calibration objects must be refitted
without their outcome or chemistry group. Refit its transformer on that inner
mean-training subset and its small predictor on the opposite distribution donor
half. Retain donor/query IDs. The final outer-query predictor can use the saved
outer-MODEL_FIT predictor. Convert scores to ranks against the corresponding
training/donor score distribution, never against query outcomes.

Fit parameters by minimizing chemistry-group-equal average fair Gamma CRPS,
not by regression on per-realization optimal scales. Bound intercept and slope
to [-log(4), log(4)] and bound each resulting increment to the same interval.
Zero is always an available candidate. Use a paired one-standard-error preference
for zero on calibration records; this is a development regularizer, not a
finite-sample risk guarantee.

Numerical fitting uses a fixed scalar-scale score grid with 17 knots and shared
4,096-draw random numbers. Interpolation is only an optimization approximation,
not a new predictive law. Re-evaluate the fitted function directly on the same
calibration distribution. If interpolation changes its mean score by more than
5e-6, refine to 33 then 65 knots; if that is still insufficient, optimize the
direct sampled objective. Record the discrepancy and solver result. This numerical
check never inspects outer-query outcomes. Always also refine the interpolated
optimum locally against the direct sampled loss: an optimum at a grid knot has
zero interpolation discrepancy but can still miss a better point between knots.

Evaluate all four arms with independent 100,000-draw common random numbers. Retain
the frozen score E[Gamma]-0.2 P(NULL), existing budgets and tie handling. Report
Gamma CRPS, NULL Brier, joint NLL, five coverage levels, predicted Gamma and NULL,
selection lists, realized value and NULL counts. Use paired chemistry-group and
layout resampling, and separate Monte Carlo error from population uncertainty.
Do not select a new primary method from these outer results.

## Interpretation and biology status

Large same-realization gains with no independent-realization benefit demonstrate
the hindsight artifact under CORE; they neither prove nor exclude missing useful
information in the real data. Direct-risk improvement would support the revised
calibration target, not biology. A null result is limited to these scalar scale
families and this data/endpoint.

The biological reference adapter remains available but disabled in the primary
method. This round does not train or enable it. Its future admission requires a
task-relevant increment over the same generic carrier and matched random-reference
controls, followed by separate-data replication. More annotations or an enabled
gate alone do not establish benefit.
