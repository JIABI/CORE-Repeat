# Same conditional kernel, geometry versus Gamma-distribution supervision

Development experiment specified before training on 14 September 2026.
The question is whether the geometric improvement can be redirected toward the
original acquisition utility without changing the response kernel or endpoint.

## Fixed model and data

Reuse the 639 opened source_5 DEV compounds, four observed roles, five compound
folds, exact TRAIN anchors and scales, frozen HR predictor, and full joint RIDGE
error covariance from conditional_response_epoch30_20260914_v1. Both new arms
use its G_CONDITIONAL_STRUCTURED architecture: 3,837 trainable parameters,
descriptor-local responses, a small conditional coefficient MLP and a bias-free
linear readout. Both start afresh at exactly the same HR prediction and parameter
state as the historical G at epoch0. The original HR MLP remains frozen.

## Two new fits and the baseline

* A_HR: copied historical predictions, no new fitting.
* J_GEOMETRY_CONTROL: reproduce the historical G training with the unchanged
  nine-coordinate MSE plus 0.1 times squared mean increment.
* K_GEOMETRY_GAMMA_CRPS: the same objective plus normalized original ADD_TWO
  Gamma CRPS, weight 1. No other model or optimization change.

The control must reproduce historical G epoch30 parameters and scored outputs.
If it does not, diagnose before interpreting the contrast. No output is chosen
by which trial happens to improve; both arms are reported at fixed actual epoch30.

## Gamma supervision

For each object, draw two independent sets of 64 standard-normal nine-vectors
per training minibatch. Multiply by the full fixed covariance Cholesky factor,
add the predicted mean, and restore the original nine-coordinate units using
TRAIN-fitted scaling. Each joint draw defines a positive triangular factor and
four virtual measurement vectors. Compute the original utility for that draw:

    Gamma = 0.5 * [cos((X+Z1+Z2)/3, V) - cos(X,V)] - 0.02.

The factor-forward differentiable calculation must agree with the existing
verified Gram decoder. It does not generate separate inconsistent cosines,
replace the endpoint, or differentiate a detached NumPy result.

For paired independent samples a and b and the observed training Gamma y, use

    CRPS estimate = mean(0.5*|a-y| + 0.5*|b-y| - 0.5*|a-b|).

The mean is over both samples and objects. This is an unbiased Monte Carlo
estimate of distributional CRPS, not a squared error of the sampled mean.
Neither antithetic pairs nor a sample paired with itself count as independent.
No additional score clipping or resampling is introduced.

Divide CRPS by the sample standard deviation (ddof=1) of the original Gamma
over the fold's fitting compounds only; reject a nonpositive/nonfinite scale.
Weight is fixed at 1, with no validation/test search. This scale conversion does
not guarantee equal gradient norms. Log the actual gradient norms and alignment
of the geometric and normalized Gamma objectives to assess their interaction.

The geometry term remains active. Improving one projected utility distribution
does not certify the entire joint measurement law or guarantee FDP control.

## Training, RNG and checkpoints

Both arms: 30 epochs, 210 updates, batch64, AdamW lr0.0003, weight decay0.0001,
gradient clipping5. Keep the original ten-update warmup and 100-epoch learning-
rate schedule, not a compressed 30-epoch schedule. Same initialization and
minibatch permutation seeds as historical G. The control calls the original
model.loss and does not add a zero-weight CRPS graph.

Use separate local generators for training Monte Carlo draws, fixed validation
draws, and fixed diagnostic draws, so none changes initialization or data order.
Validation uses 128 independent pairs per object, fixed across epochs and arms.
Every five epochs record fit/validation geometric MSE, fixed-draw Gamma CRPS,
Gamma mean MSE, loss components, branch gradient norms and their cosine. Save
epoch0/5/.../30 with optimizer, schedule and RNG state. Validation-best geometric
checkpoint is supplementary; actual epoch30 is the comparison endpoint.

## Evaluation

Apply the unchanged verified decoder with 2,000 joint samples per object and the
same evaluation seeds as the historical experiment. Report geometry MSE, Gamma
CRPS and mean MSE, NULL Brier, ranking, joint and marginal coverage, and actual
budget value with FDP and false-activation risk. At 25% additional physical-well
budget, select 16/16/16/16/15 objects across folds, total79 objects and158 wells.
No purity criterion is removed and no cost is deducted twice.

Primary comparisons K−J and K−A use 2,000 paired within-fold compound bootstrap
resamples of fixed predictions. They quantify this development comparison, not
training/batch uncertainty or an independent certificate. Reduced Gamma CRPS
with unchanged selection is a distributional improvement only; improved value
with increased risk must be reported as a tradeoff.

No further training beyond30 is started automatically. Original outcomes,
contracts, split files and old runs remain intact; FINAL400, fifth repeats and
unopened candidates remain unopened, and old paused training stays paused.
