# A covariance-matched ordinary/rare-large residual law

## Question and fixed scope

Does a shared radial scale mixture improve the conditional error distribution
when the STATE50 geometry mean and LOCAL_FIT covariance are unchanged?
This is a retrospective development comparison on the same opened 1,188 LINCS
objects, not new experimental data or a confirmatory test.

Reuse the ten saved FIT/CAL/query cells, fitting-reference covariance choices,
and 40 ID-min calibration chemistry-group representatives per cell from
`runs/lincs_joint_residual_borrowing_20260916_v1`. Load the saved LOCAL_FIT
covariances directly. There is no model training, covariance refit, reference
resampling, biological input, FINAL access, or additional well access. Only the
declared distribution candidate is chosen using calibration outcomes.

## Two predictive arms

**GAUSSIAN** is the saved LOCAL_FIT law, N(mean_i, C_i).

**RADIAL_SCALE** selects one of the following five candidates using mean full
nine-dimensional calibration NLL. The candidate order also fixes ties:

1. Gaussian (V=1).
2. v_small=0.5, p_large=0.1, v_large=5.5.
3. v_small=0.5, p_large=0.2, v_large=3.
4. v_small=0.75, p_large=0.1, v_large=3.25.
5. v_small=0.75, p_large=0.2, v_large=2.

The mixture draws V=v_large with probability p_large and V=v_small otherwise,
then draws one complete residual vector

    e_i = sqrt(V) chol(C_i) z,  z ~ N(0,I_9).

One V is shared by all nine coordinates of a draw. Since
v_large=[1-(1-p_large)v_small]/p_large, E[V]=1. Thus E[e_i]=0 and Cov(e_i)=C_i
exactly; no rescaling after calibration or covariance clipping is used.
The law changes radial tails and higher-order dependence, not skew or
direction-dependent shape. It is a conditional geometry-error model, not a
decomposition into physical common and independent well noise.

The originally considered Student-t family is deferred: a nondegenerate
Student-t in log-Cholesky coordinates has no exponential moments, which can
make physical Gram moments undefined. A finite Gaussian mixture has all
exponential moments while retaining an ordinary/rare-large interpretation.

## Calibration and regions

Use only the same 40 calibration representatives' residuals and covariances.
Evaluate the exact mixture density by logsumexp, including each component's
9 log(V) determinant term. Pick the lowest average NLL. Differences within
absolute 1e-12 plus relative 1e-12 count as ties and retain the earlier candidate,
so ties favor Gaussian. Keep all candidate scores; do not refit after selection.

For each nominal level in {0.5,0.8,0.9,0.95,0.99}, solve the analytic joint-radius CDF

    (1-p) F_chi2_9(q/v_small) + p F_chi2_9(q/v_large) = level.

The coordinate central interval is mean_j +/- sqrt(C_jj) h, where

    (1-p) Phi(h/sqrt(v_small)) + p Phi(h/sqrt(v_large)) = (1+level)/2.

These are model-based regions, not conformal or formal coverage guarantees.
The covariance ellipsoid orientation and marginal variance stay fixed even
though its probability-calibrated radius can change. Nine-dimensional u-space
regions are not regions in the physical profile or Gram-entry metric.

## Scoring and saved outputs

Use 10,000 draws per query and the existing observable forward map and metrics.
The normal RNG seed is 20260916+100*fold+half, in the same 16-query blocks as the
previous Gaussian scorer. An independent RNG at seed+36000 selects the scalar
component. Gaussian selection reuses the exact Gaussian scores and draws.
All transformed samples must be finite; no sample is clipped, removed or
resampled. Save per-object results, candidate choices, analytic quantiles and
moment checks. Verify Gaussian replay against all existing saved score arrays
and exact mean/covariance identity across the two arms.

Report full NLL, energy, coordinate/joint coverage at the five levels, all
single/difference/average observable scores, absolute pair differences, Gamma
CRPS/coverage, NULL Brier, rank correlation, acquired value and NULL counts.
The established primary coverage fields stay at 95%; 99% is an additional
upper-tail diagnostic, not a replacement for the primary level.
Preserve each query-cell budget and the total 146-object/292-well allocation;
rank by predicted Gamma with deterministic ID tie-breaking. Reference costs
remain excluded, as in the preceding diagnostic.

The primary contrast is RADIAL_SCALE minus GAUSSIAN. Include all fold and cell
directions and fixed-prediction chemistry/layout bootstrap intervals; these do
not cover refitting or sequential development selection. A distribution score
gain without allocation gain supports distribution modeling, not better
acquisition. Small or absent policy change is possible, not guaranteed.
Gaussian selection is a complete allowed outcome and does not trigger another
candidate search. This plan must be reviewed before the new query run begins.
