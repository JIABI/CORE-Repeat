# C3: conditional direct distributions of follow-up value

## Question and scope

Does the distribution-score advantage of CORE persist against a direct model
whose conditional distribution can change shape and scale with the initial
measurement? The previous direct comparator predicted a mean and used a pooled
calibration-residual distribution. C3 adds conditional-quantile HistGB, without
changing CORE, the measurement task, the two-well action, or its declared cost.
This is a development-data comparison, specified before fitting the new arm;
it is not a new independent confirmation or a revision of R4.

## Data and access

Use all saved R2 deployment cells: EU (904 objects, 5 cells), JUMP (639, 5),
LINCS (1,188, 10), and RxRx3 (10,410 compound-dose conditions, 40). Reuse the
original ACCESS_MATCHED HistGB input features, saved preprocessing, TRAIN plus
REF_FIT training rows, VALIDATION rows, DIST_CAL rows, DEV_EVAL rows, and their
ordering. All doses of a chemical group retain their original outer-fold role.
No R4 input, outcome, fitted model, selection list, or policy is changed.

Training and calibration outcomes are the saved realized Gamma values for the
original task. Original CORE and direct HistGB predictions are reused by exact
object identity. HistGB's separately calibrated classifier remains its original
decision comparator; its residual-law CRPS is identified separately from that
classifier's probability.

## Full method configuration

Display name: Conditional-quantile HistGB. Implementation: scikit-learn
HistGradientBoostingRegressor, installed release 1.9.1, quantile (pinball) loss.
The user authorized this full four-dataset experiment on 22 September 2026.

Fit 19 quantiles: 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.45,
0.50, 0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, and 0.99. Every fit
uses 200 boosting iterations, learning rate 0.05, L2 regularization 1,
max_bins 255, max_features 1, no maximum depth, and no early stopping. The
three candidate (max_leaf_nodes, min_samples_leaf) settings are (7,20),
(15,10), and (31,10), matching the previous HistGB search. Use the saved cell
seed, recorded in each new cell manifest, for all its quantiles and settings.

Choose a single setting shared by all 19 quantiles using mean VALIDATION CRPS
of the reconstructed law. Ties within absolute/relative tolerance 1e-12 keep
the first, simpler setting. Retain that TRAIN+REF_FIT fit: do not refit on
VALIDATION or DIST_CAL. This gives 3,420 full quantile fits across 60 cells.

## One coherent scalar predictive law

For each object, sort quantile predictions to remove crossings, then clip to
the physical Gamma range [-1.02, 0.98]. Append p=0 and p=1 endpoints by linear
extrapolation from the nearest two extreme quantiles, clipped to the same
bounds. Interpolate the quantile function linearly in probability. Flat
segments are point masses; P(NULL) includes any mass exactly at zero.

Compute expected Gamma, P(Gamma <= 0), CRPS, and central 50/80/90/95/99 percent
intervals from this same law. Means and CRPS are analytic integrals, not new
Monte Carlo estimates. These one-dimensional distributions do not predict
the joint future wells; no joint-well NLL or joint coverage is attributed to
them. Raw CDF values at observations are not treated as continuous PIT values
when the reconstructed law has atoms.

The raw conditional-quantile law is the primary new arm. A prespecified
secondary arm adds a DIST_CAL-only offset to each quantile: the empirical
alpha-quantile of y_cal minus the raw prediction at alpha, using linear sample
quantile interpolation. Apply offsets to raw query predictions before the
same rearrangement and tail construction. This arm is reported separately,
not selected against the primary arm using query performance.

## Single-pass evaluation and reuse

Report Gamma CRPS, mean squared error, NULL Brier/AUC, interval coverage and
width, and calibration summaries. Use each original cell's fixed quota, rank
by E[Gamma] - 0.2 P(NULL), and break ties by object identity. Report total and
mean realized Gamma, NULL count/rate, forecast NULL count, selected-region
Brier, selection overlap, per-cell risk, and budget curves. Lambda=0 is a
secondary policy readout. Keep the original CORE/HistGB selection lists and
also check agreement with deterministic reconstruction. The fixed-budget
random reference is the exact cell-quota-matched expectation, not a single
random draw or the constant-score policy.

Use 2,000 paired resamples of chemical groups and of saved layouts for
conditional-on-fitted-model comparisons. Repeated doses remain within their
chemical group. Report the two dependence analyses distinctly and the number
of actual groups/layouts; do not treat 60 cells as independent trials. These
are development comparisons, not equivalence tests or finite-sample risk
guarantees. Save all per-object distributions, scores, identities, selections,
and timings for later paper/R1-R3 reuse without rerunning model fitting.

## Execution and interpretation

Run two CPU worker processes, each with one numerical thread. Checkpoint every
completed quantile fit and cell so restart does not repeat completed work.
The first complete cell is part of the final comparison and is used to update
the runtime estimate. Save training, prediction, and integration timings
separately. Historical CORE fitting and sampling costs are not zero merely
because its predictions are reused here.

If the CRPS gap closes, narrow the previous claim to the benefit of conditional
distribution modelling over the tested pooled-residual baseline. If the gap
persists, report superiority over this stronger direct distribution model;
do not infer from C3 alone that joint dependence caused it. Allocation and
selected-risk results are evaluated separately in either case. Do not change
the architecture, quantile grid, tail rule, or primary arm after reading query
results. Cross-site policy transfer is outside this experiment.
