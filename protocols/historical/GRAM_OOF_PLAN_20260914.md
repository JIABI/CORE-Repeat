# Paired development cross-validation of acquisition-value distributions

## Question and data

Compare GLOBAL_GEOMETRY, RIDGE_GEOMETRY, L_GRAM and G_DIRECT on the same
out-of-fold objects. The question is whether conditioning on X improves the
distribution of the original acquisition utility and fixed-budget decisions.
Only the already opened source5_primary_fullcontrols export is used: 639
compounds, X/Z1/Z2/V, all 3,617 coordinates. The original split files, FINAL,
fifth repeat, additional candidate pool, outcomes, costs and seven-item contract
are unchanged. This is a new analysis of reused DEV, not a new blind cohort.

## Folds fixed before model fitting

Seed 20260914. Assign all compounds to five outer folds, stratified by ten
equal-count ranks of log ||X|| (ID breaks exact ties). Neither future outcomes
nor previously observed errors define strata. Each outer training pool is split
80/20 into fit and inner validation, with the same X strata. All four models
fit parameters and preprocessing on the same fit subset (about 409 objects).
G alone uses the inner validation labels for checkpoint selection; the other
models retain their declared training-only estimation procedures. Thus parameter
fit objects match, but selection-label use is not identical across methods.

Every compound has exactly one outer-test prediction. No original in-sample
prediction is counted as out of fold. Every affine scale, PCA basis, covariance,
target-coordinate scale and model fit is reconstructed within the proper fold.
Future profiles are used as training targets, never as query inputs. These folds
evaluate generalization across compounds under shared batches, not new batches.

## Complete model configurations

* GLOBAL: training mean and full Ledoit-Wolf covariance in the existing nine
  log-Cholesky coordinates of WW^T/||X||^2. One shared integration stream is
  broadcast within each test fold, yielding exact ties across its objects.
* RIDGE: all 3,617 affine-scaled X coordinates plus log raw X norm, unpenalized
  intercept; penalties 0.01, 0.1, 1, 10, 100 with alpha=n_fit*lambda. Final
  penalty uses five-fold internal CV. Predictive-error covariance uses separate
  outer-five/inner-four nested predictions. Each internal fit reconstructs its
  input and target scales. Penalties compare errors in that internal fit's
  standardized target coordinates. Residual predictions are returned to native
  u and then the enclosing fit scale before the full covariance is estimated as
  LedoitWolf(centered errors) + error_bias outer product. Do not add that bias
  again to the mean. Numerical penalty ties prefer the larger penalty.
* L: refit the existing full-coordinate conditional Gaussian moment model on
  raw fit profiles, rank 200, fit clipping 8, noise shrinkage 0.05, variance
  floor 1e-6; retain its independent full-coordinate residual model. Sample
  complete future profiles and then form the original Gram. No old anchor is
  reused and no output coordinate is omitted.
* G: unchanged grouped-profile encoder, width 64, two attention layers/four
  heads; mean MSE and detached-mean covariance NLL, full joint covariance with
  0.05 identity shrinkage/log-standard-deviation bound 4. AdamW, learning rate
  0.0003, weight decay 0.0001, batch 64, clip gradients at 5; 30-step warmup and
  cosine decay to 0.000003 over 200 maximum epochs. Evaluate inner-validation
  ADD_TWO Gamma CRPS with 256 fixed-stream samples every five epochs. Keep the
  minimum-CRPS checkpoint; stop after eight checks without improvement of
  0.00001, no earlier than epoch 40, or at epoch 200. Report every 20 epochs.

Each fold uses seed 20260914 + 10000*fold for fits and separate fixed streams
for selection and test sampling. No JEPA, new biological kernel or mechanisms
branch is added. This comparison tests complete existing model families, not a
single-factor output-dimension ablation.

## Forward geometry and numerical records

All direct-coordinate models use the already tested factor-forward assembly
of [1,0,0,0] and [p,L]. This computes the same PSD Gram law without requiring
subtraction of nearly equal terms to recover its Schur factor. Record the
original strict inverse-recovery failures alongside every validation/test
evaluation. Check each test draw against direct virtual-vector utilities.
No clipping, jitter, rejected draws or excluded objects are introduced. A direct
factor/forward-utility failure remains a numerical failure, not a missing row.
This new prospective numerical route does not reclassify old stopped runs.

## Test evaluation and reporting

2,000 complete joint samples per object, float64 geometry, original ADD_ONE
cost 0.01 and ADD_TWO cost 0.02. Retain joint draws, original gains, probability
scores, geometry observables, point predictions and numerical diagnostics.
The same fit-derived geometry scoring scale is used for every arm in a fold.

Primary comparison: G versus GLOBAL ADD_TWO Gamma CRPS. Report all six paired
arm comparisons, NULL Brier/AUC, gain bias and rank association, and all original
5/10/25% within-action and physical-well budgets. The predeclared principal
policy readout is ADD_TWO expected-gain ranking at the 25% physical-well cap.
Report value and FDP/FPR together, not a retrospective choice of best budget.

Execute selection separately within outer folds, then sum the actual allocated
wells, selected objects and gains. GLOBAL uses the uniform-subset expectation,
not ID-selected outcomes. Never rank its different fold-specific constants as
individual information. Report fold results, object-weighted scores and
within-fold rank association. Overall conditional-model ranks are descriptive
because the five fitted rules differ. Pairwise bootstrap intervals resample
compounds within outer folds with masks fixed. They condition on fitted models
and shared batches and do not account for overlapping training sets, historical
development search or new-batch variation. No formal certification is issued.

The existing-prediction eight-dimensional geometry diagnostic and fixed
X-neighbor borrowing diagnostic run separately. Neither changes this model
configuration in response to its results. Any later 8D retraining or structured
kernel experiment requires a separately specified comparison.
