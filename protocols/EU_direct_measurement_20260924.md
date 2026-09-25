# EU direct measurement prediction comparison

Specified on 24 September 2026 before fitting the new comparator. The author
authorised this experiment before manuscript changes. This is a post-hoc
method comparison on already opened confirmation measurements, not another
independent confirmation. The original R4 models, predictions, 192-object
lists, endpoints and comparisons remain unchanged. No protected JUMP data,
new site or new measurement is accessed.

## Question and fixed endpoints

Compare the frozen CORE forecast with conditional-quantile HistGB trained
directly on the three existing transformed two-future-well-average norms:

T_ab = log(1 + ||(Y_a + Y_b)/2||^2 / ||X||^2),

for (a,b) = (Z1,Z2), (Z1,V), (Z2,V). These are the existing measurement
observable indices 6, 7 and 8; none contains X in its numerator. The primary
metric is the per-object mean of their three CRPS values, averaged equally
over the common complete evaluation objects. The primary contrast is CAL
HistGB minus CORE (positive favours CORE). Report RAW separately without
selecting the preferable version on query outcomes.

## Data, roles and predictors

Use the existing EU final-model partition of the 904 development identities:
TRAIN 434, VALIDATION 108, REF_FIT 181, DIST_CAL 181. Training of direct
models uses TRAIN followed by REF_FIT, 615 identities. Validation selects
hyperparameters; DIST_CAL fits only the stated offsets. No refitting on
validation or calibration and no query outcome enters either process.
Reuse the frozen TRAIN-derived input transformation and the full initial
profile, log-norm and permitted chemical descriptor inputs of R4 HistGB.
The input is not reduced to a few summary features.

Query prediction retains the frozen order of all 1,527 eligible initial
profiles. Evaluation subsequently uses the same 1,520 complete four-well
objects, independently of selection. The 19 incompletely observed identities
are not assigned values for these unbounded norm endpoints; this analysis
estimates predictive performance on the complete-observation population.
Original missing-outcome bounds for the declared gain remain unchanged.

## Direct models and distribution construction

Use scikit-learn 1.9.1 HistGradientBoostingRegressor with quantile loss.
Fit levels 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.45, 0.50,
0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975 and 0.99.
For each of the three targets, compare (max_leaf_nodes,min_samples_leaf)
(7,20), (15,10), (31,10). Every model uses 200 iterations, learning rate
0.05, L2 regularization 1, max_bins 255, max_features 1, no depth limit,
no early stopping and seed 20260921. There are 171 complete fits.

Choose one candidate per target, shared across its 19 quantiles, by mean
VALIDATION CRPS of the RAW reconstructed law. Absolute/relative ties within
1e-12 retain the earlier candidate. Save all candidates and their validation
scores. Model selection does not use CAL or confirmation outcomes.

RAW sorts each row of quantile predictions, floors them at zero, and linearly
extrapolates the adjacent extreme knots to probabilities zero and one.
The lower endpoint is floored at zero; the upper endpoint is not clipped.
Between knots the quantile function is linear, retaining atoms on flat
segments. The finite extrapolated upper endpoint is a numerical predictive
approximation, not a physical upper bound on T. No outcome-derived maximum,
Gamma bounds or transformation fitted on confirmation is used.

CAL uses the same selected models. At level alpha its offset is the linear
sample alpha-quantile of T_cal minus the unrearranged alpha prediction.
Add offsets before sorting, zero flooring and the identical tail rule.
CAL is the primary access-matched comparator because CORE uses DIST_CAL
labels for its radial law. RAW is a mandatory secondary comparator. This
post-processing has no distribution-free coverage guarantee.

Means and CRPS are evaluated by exact integration of each piecewise-linear
quantile law. Central intervals use levels 50, 80, 90, 95 and 99 percent.

## Frozen CORE replay and reusable outputs

Reuse cached geometric means/scatters, coordinate preprocessing, empirical
radial law and query-specific weights in r4_confirmation_20260921_v1. Do not
refit or rerun the mean model. Replay 100,000 draws per eligible object with
the original primary seed 20260921, radius seed plus 47000, chunk size eight,
and all 1,527 eligible objects in their original order. Filter to the 1,520
complete objects only for scoring. Compare replayed Gamma against the
existing float64 cache to verify that the original stream was retained.

Use the existing factor-forward observable implementation. Save predictions,
fair Monte Carlo CRPS, five-level intervals, widths, coverage, means and MC
mean errors for all ten existing measurement observables in the same pass;
only the stated three define this comparison. No new acquisition policy is
created. No frozen selection list is rewritten. Full Monte Carlo tensors
need not be stored: retain per-object summaries and the replay specification.

## Statistics, computation and reporting

Report all three targets and their equal-weight family mean for CORE, RAW
and CAL. Pair on identical object IDs. Use 10,000 paired resamples of chemical
identity groups and, separately, the seven library layouts, retaining all
objects in a sampled block and weighting objects equally. Seed 20260924.
Percentile 95 percent intervals are conditional on these fitted models;
the layout analysis is a dependence sensitivity with seven blocks, not a
new-environment validation. The family-mean CAL-minus-CORE CRPS comparison
is primary. Individual targets, RAW, coverage and width are supporting
diagnostics, not additional independent discoveries. Save absolute and
relative score contrasts, paired coverage/width differences, per-layout
summaries, all per-object scoring arrays, selected validation configurations
and timing records. Relative CORE improvement uses the direct-model score
as denominator and is recalculated in each resample.

Use at most two CPU worker processes, one numerical thread per worker.
Checkpoint completed quantile fits and the CORE replay. Reuse completed
outputs on restart. Keep fit, prediction, calibration and replay timings
distinct; reused historical CORE fitting costs are not zero. Existing
geometry equivalence tests and the new nonnegative-law tests check changed
paths; they are not scientific results. No parameter, target, tail or
primary-arm changes will be made after observing query performance.

If CORE has lower CRPS, report its advantage for these observables and this
cohort. If differences are unresolved, report that without claiming
equivalence. If direct prediction is better, report the performance trade-off.
This comparison does not isolate a single component; the existing
matched-marginal scatter ablation supplies that separate evidence.
