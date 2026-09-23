# Borrowing complete conditional residual records

## Question and scope

Can the distribution of complete reference residuals improve the shape of
conditional joint-error predictions beyond a Gaussian with the same mean and
covariance? Does local retrieval help beyond population borrowing, and do any
distribution improvements reach Gamma, NULL probability and acquired value?

Use only the opened LINCS Cell Painting 1,188 objects and four roles. Preserve
STATE50 means, the existing five outer folds, ten query/reference cells, and the
fit/calibration reference split from `lincs_joint_tail_calibration_20260916_v1`.
Keep the original Gamma, NULL definition and the 146-object/292-query-well
allocation. No FINAL, fifth replicate, biological annotations or JEPA are used.

## Fixed conditional location and covariance

Reproduce the previous LOCAL_FIT covariance procedure exactly: original frozen
fold covariance; reference residuals around frozen STATE50 means; current
direction/Morgan/amplitude retrieval; top16; beta selected from {0,.25,.5,.75}
by fitting-reference leave-chemical-group-out Gaussian NLL. Neither calibration
nor query outcomes fit these covariances. The query mean and covariance are
identical in every arm of this new experiment.

For fitting reference j, standardize its full residual by the Cholesky factor
of its leave-chemical-group-out predicted covariance:

    h_j = L_j^-1 (u_j - mean_j).

The covariance strength beta is selected on fitting references, so these are
training-reference LOO residual scales, not a new independent validation set.
The raw means were already held out from the frozen outer-fold model.

## Complete continuous residual law

Let w_ij be feature-only reference weights, m_i=sum_j w_ij h_j and
S_i=sum_j w_ij (h_j-m_i)(h_j-m_i)'. Fix smoothing bandwidth b=0.5 before results.
Define V_i=(1-b²)S_i+b²I and B_i=chol(V_i). Draw one complete reference index J
from w_i and one standard Gaussian vector z. The residual is

    e_i = L_i B_i^-1 [sqrt(1-b²)(h_J-m_i) + b z].

This is a full-density Gaussian mixture, not independent resampling of each
coordinate or physical well. It has exactly E[e_i]=0 and Cov(e_i)=C_i; no
eigenvalue clipping or outlier removal is needed because b>0. Full NLL is
computed by stable mixture-density evaluation, not a Gaussian surrogate.

Centering and covariance matching keep the mean/covariance comparison fixed.
They deliberately do not test whether reference residual bias or changed
second moments would improve the model. Shape includes marginal tails/skew and
cross-coordinate higher-order dependence, not just biological shared noise.

Shrink this complete residual law toward N(0,C_i) with mixture weight alpha in
{0,.25,.5,.75,1}. Choose alpha by full NLL on the 40 ID-min calibration chemical
group representatives from the previous split; ties favor smaller alpha.
Calibration outcomes choose this one strength only, not reference identities,
bandwidth, smoothing or mean/covariance. Do not refit with calibration labels.

## Four arms

1. **GAUSSIAN:** the previous LOCAL_FIT law, unchanged.
2. **GLOBAL_BLOCK:** uniform fitting-reference weights, calibration-selected alpha.
3. **LOCAL_MATCHED:** weights equal to half uniform plus half current top16 local
   weights; use GLOBAL_BLOCK's alpha unchanged.
4. **LOCAL_BLOCK:** the same local weights, with its own calibration-selected alpha.

GLOBAL_BLOCK versus GAUSSIAN tests population residual shape at fixed moments.
LOCAL_MATCHED versus GLOBAL_BLOCK changes retrieval only, conditional on the
chosen global strength. LOCAL_BLOCK versus GLOBAL_BLOCK compares two complete
calibrated procedures; it does not isolate retrieval from strength selection.
The half-uniform component supports global fallback when a local neighborhood
is small. It is fixed rather than optimized on queries.

## Scoring and reproducibility

Every query is evaluated once. Use 10,000 independent draws, common normal and
uniform random streams across arms. If alpha is zero, return the Gaussian law
exactly, including its evaluation draws and analytic regions. Store residual
library, all weights, component centers/covariances, calibration candidate
scores and chosen alphas so densities and samples can be independently rebuilt.

For non-Gaussian arms, use each predictive law's own 95% squared Mahalanobis
quantile to define its joint ellipsoid; do not use a chi-square threshold merely
because the covariance is matched. Estimate it from model samples, not query
outcomes. Coordinate intervals use marginal sample quantiles. These are
model-based empirical coverage checks, not split-conformal certificates.

Report NLL, energy score, mean and covariance identity checks, joint/coordinate
coverage and interval width, all previous single/difference/average observables,
absolute pair differences, Gamma CRPS/coverage, NULL Brier, Gamma ranking,
selected actual value and NULL count. Retain per-cell budgets and rank within
query cells; auxiliary reference acquisition costs remain excluded.

Primary comparisons: GLOBAL_BLOCK−GAUSSIAN, LOCAL_MATCHED−GLOBAL_BLOCK,
LOCAL_BLOCK−GLOBAL_BLOCK, LOCAL_BLOCK−GAUSSIAN. Paired fixed-prediction chemistry
and layout bootstrap intervals are development diagnostics and exclude refitting,
reference sampling uncertainty and repeated-development selection. Show all fold
directions and alpha choices; do not promote an arm using pooled query outcomes.

## Decision rule for interpreting this experiment

- Better proper scores without better acquisition value support measurement
  distribution modeling, not improved allocation.
- Improved marginal coverage alone does not establish correct joint geometry.
- Alpha returning to zero is an allowed result, not a reason to expand a search.
- Failure of this finite reference-mixture family does not disprove every
  non-Gaussian or biological model; sample support and shape assumptions matter.
- Only after this comparison should target/MoA retrieval be introduced as a
  separate switch. Representation learning remains a separate later comparison.
