# Conditional measurement geometry: linear backbone and joint-error repair

## Question and scope

Test the proposed linear-plus-constrained-neural mean and out-of-fold joint
error estimation on the original acquisition utility. This is the fixed-role
geometry implementation of that estimator. It does not identify separate
source, batch, plate or biological variance components, and is not a complete
cross-site hierarchical biological world model. All nine legal Gram coordinates
are retained, with all 3,617 X coordinates and X log norm as input.

Use exactly the 639 opened DEV compounds, four roles, and five outer
fit/inner-validation/test assignments from gram_oof_20260914_v1. No new
partition, fifth replicate or FINAL is opened. Costs, outcomes, original
seven-item contract, original split files and all old results remain unchanged.
This is further development on reused data, not new independent validation.

## Seven arms

Reuse the complete GLOBAL_GEOMETRY, RIDGE_GEOMETRY, G_DIRECT and L_GRAM results
from the preceding comparison, without recomputing or replacing their values.
Three new arms isolate the intended changes:

1. G_OOF_COV: the exact stored best G mean and preprocessing; replace its entire
   covariance with the full second moment of nested out-of-fold mean errors.
2. HR_RIDGE_COV: the original fitted RIDGE mean plus a bounded residual network;
   retain the exact original RIDGE covariance. This isolates mean correction.
3. HR_OOF_COV: the same corrected mean, with its own full out-of-fold joint
   error covariance. This is the combined estimator under test.

G's learned covariance is replaced, not added to. Likewise HR_OOF_COV replaces
RIDGE covariance rather than adding a second uncertainty component. Error
covariance contains prediction error and bias as well as measurement variation.

## Mean and fitting

HR uses an immutable ridge coefficient/intercept and a fully connected residual
network D+1 -> 32 -> 32 -> 9, GELU, dropout 0.1 between hidden layers, and a
zero-initialized last layer. Correction = 0.5*tanh(network output), in the
training-standardized nine-coordinate frame. Loss is mean squared error plus
0.1 times mean squared correction. No covariance loss updates the mean.
The network uses float64 to retain the baseline coefficients exactly.

AdamW, learning rate 0.0003, weight decay 0.0001, batch 64, gradient clipping
5, 30-step warmup, cosine decay to 0.000003, at most 200 epochs. Check mean
MSE on the same 103 external inner-validation compounds every five epochs,
including epoch zero. Save the minimum-MSE checkpoint, stop after eight checks
without a 0.00001 improvement, no earlier than epoch 40. Log every epoch and
save a report every 20 epochs. Epoch-zero selection is allowed and reported.
The correction bound and penalty are design choices, not established optimal
values; neither is searched in this round.

## Out-of-fold errors

Use the original RIDGE's five internal error-fold memberships and four-fold
inner-selected penalties. Reconstruct each internal ridge from its fitting
objects only, including preprocessing, and verify that its predictions restore
the saved native-coordinate OOF predictions. Thus zero neural correction
reproduces the original RIDGE mean-error estimator without changing ridge CV.

For each internal fold independently fit HR and G using only that fold's
training compounds. Each fit reconstructs all preprocessing from those objects.
The same outer inner-validation set is used for checkpoint selection and is
disjoint from both internal fitting and error-holdout objects. HR selects on
coordinate MSE. G clones retain the complete original G recipe and select on
original ADD_TWO Gamma CRPS; the final G mean is never retrained or reselected.
Do not use the final model's in-sample errors as OOF residuals.

Restore each held-out prediction to native u, then express all errors in the
enclosing fit-only standardized frame. Covariance is
LedoitWolf(centered errors) + outer(error mean, error mean), exactly the original
RIDGE convention. No additional correction is applied to the predictive mean.
Save split IDs, predictions, errors, selected epochs and covariance components.
This nesting excludes each error object's outcomes from fitting and selection,
but overlapping training sets/shared validation/batches remain development
dependencies rather than independent certification units.

## Evaluation

Generate 2,000 full joint samples per outer-test compound with shared integration
seeds for the two HR covariance arms. Use the existing factor-forward legal Gram
decoder, original Gamma and original policy evaluator. No outcome clipping,
object removal, rejection sampling, jitter or outcome-driven stratification.

Primary score comparison: HR_OOF_COV minus RIDGE_GEOMETRY, ADD_TWO Gamma CRPS.
Mechanism comparisons: HR_RIDGE_COV minus RIDGE_GEOMETRY (mean),
HR_OOF_COV minus HR_RIDGE_COV (covariance), and G_OOF_COV minus G_DIRECT
(covariance with a fixed G mean). Also retain GLOBAL and L in every summary.
Report NULL Brier/AUC, Gamma bias/rank association, geometry energy, norm/angle/
well-difference/well-average predictive coverage, and all three actions.

The principal policy remains ADD_TWO expected-gain ranking at the 25% physical
well cap, selected within folds: 79 compounds/158 wells in aggregate. Report
actual value and FDP/FPR together, alongside all original 5/10/25% budgets.
No post hoc best budget, risk relaxation or certification claim.

Additionally save analytical u means/covariances and actual coordinates for
the Gaussian-coordinate models, including old G/RIDGE/GLOBAL. Report MSE,
R-squared against the fit mean, bias, error concentration, Gaussian marginal
and joint-ellipsoid coverage, and whitened error moments. These are diagnostics
of predictive error, not identified physical variance. L's transformed law is
not assumed Gaussian and receives no chi-square ellipsoid claim.

Paired intervals resample compounds within outer folds with fitted predictions
and policy masks fixed. They do not include historical model search, training
overlap or shared-batch uncertainty. Both favorable and unfavorable results
are reported without selecting a new winner for FINAL.
