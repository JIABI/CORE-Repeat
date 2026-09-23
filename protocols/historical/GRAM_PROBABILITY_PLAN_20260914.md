# Conditional Gram probability: first development comparison

## Question and scope

Does predicting a legal joint geometry, rather than reconstructing all future
profile coordinates, improve the distribution of the original acquisition
utility? This experiment does not select a JEPA teacher. It implements a direct
conditional probabilistic model with a fixed observed target.

Only the existing `source5_primary_fullcontrols` export is used: 639 previously
opened DEV compounds, four existing roles X/Z1/Z2/V, 3,617 coordinates. The
historical split remains TRAIN 383, validation 96, calibration 64, evaluation 96.
The original FINAL, fifth repeat, other candidate pools, outcomes, and seven-item
contract are not changed. No compound is removed after inspecting errors.

## Arms and information

- **L_GRAM**: the existing full-coordinate conditional Gaussian L, loaded from
  its saved moment-fit anchor, with joint samples mapped into Gram matrices.
  No refitting is performed.
- **J10_GRAM / F10_GRAM**: the saved, validation-mean-selected epoch-10 J/F
  checkpoints, using their original implementation, scaler, chemistry,
  reference, library context, and joint sampling. These are historical
  contextual comparators, not information-matched output-only ablations.
- **G_DIRECT**: a new complete conditional nine-dimensional Gaussian density in
  log-Cholesky geometry coordinates. It uses the entire X and its deterministic
  log-norm descriptor, without JEPA, new kernels, chemistry or reference input.
  Its decision-time information matches L. Its architecture also differs from
  L, so an improvement is not attributed exclusively to output dimension.

No claim of an output-only causal ablation between G and the old neural arms
will be made. A matched full-spectrum neural retraining, if warranted by this
development result, is a separate follow-up experiment.

## Fixed target and legal distribution

Compute G = W W^T / s in the original declared measurement space, with
W=(X,Z1,Z2,V) and s=||X||^2 known at decision time. This common positive scaling
preserves every declared cosine utility. It retains future amplitudes relative
to X, and is not endpoint whitening or PCA truncation.

For G00=1, let p=G[1:,0] and H=G[1:,1:]-p p^T. Factor H=L L^T and define

u = (p0,p1,p2,log L00,L10,log L11,L20,L21,log L22).

Joint samples of all nine coordinates are transformed back to G. H is positive
definite in the model interior; singular empirical boundaries are checked and
reported, not silently perturbed. Zero-norm/nonfinite measurements are not
silently assigned cosine zero. The existing complete cohort must pass this
representation check before execution.

The density is on standardized u, not on nine independent cosines. Original
Gamma is evaluated sample by sample; Gamma(E[G]) is never reported as E[Gamma].
The raw four-well utility and the Gram utility must agree numerically.

## G_DIRECT model and training

- Reuse the coordinate-complete CellProfiler grouped encoder: hidden width 64,
  two inter-group attention layers, four attention heads; no coordinate subset.
- X uses the existing TRAIN-fitted invertible affine scaler. The log original
  X norm is standardized on TRAIN. Fixed role order is part of the task.
- u center and scale are fitted on TRAIN only. No endpoint is clipped.
- The mean encoder/head is trained with MSE in standardized u.
- The covariance head uses detached mean features and detached mean predictions
  in its Gaussian NLL, so covariance gradients do not change the mean branch.
  This protects the mean in u coordinates, not a claimed guarantee about E[G]
  or E[Gamma] after nonlinear transformation.
- Covariance is full: Sigma(x)=0.95 D(x) R(x) D(x)+0.05 I. R is a legal learned
  correlation matrix and log standard deviations are bounded to [-4,4].
  Initialization is identity covariance. This fixed weak shrinkage is not tuned
  against held-out outcomes.
- Separate AdamW optimizers, learning rate 0.0003, weight decay 0.0001,
  gradient clipping 5, batch size 64. Warmup 30 optimizer steps, cosine decay
  to 0.000003 over a maximum of 200 epochs. Seed 20260914. CPU execution.
- Every five epochs, use 256 fixed-stream joint draws on validation and select
  by original ADD_TWO Gamma CRPS. Training and validation random streams are
  separate. No FDP-based checkpoint or budget selection.
- Stop after 8 validation checks without improvement of at least 0.00001,
  after a minimum of 40 epochs; otherwise stop at epoch 200. Keep last and best
  checkpoints. The selection rule is unchanged after seeing results.
- Save a compact progress report every 20 epochs. Repeated cross-validation or
  more seeds are not silently added in this first diagnostic run.

## Common evaluation

The evaluator is fixed before comparison. Each saved model produces 2,000
joint draws per object on the three original non-TRAIN DEV partitions.
Chunking controls memory but does not replace the distribution.

1. Geometry: six pairwise cosines, relative well norms, future-well difference
   energies, pair-average energies, and acquisition-average energies. Report
   predictive coverage/width, CRPS, bias, and object-level errors.
2. A common joint energy score on the nine free entries of the normalized Gram,
   using one TRAIN-fitted scale shared by all arms. It is not a comparison of
   incompatible full-spectrum and nine-dimensional NLL values.
3. Utility: mean bias, spread, Gamma CRPS, NULL Brier/AUC/calibration and rank
   correlation. ADD_ONE costs 0.01 and ADD_TWO costs 0.02 are unchanged.
4. Policy: existing within-action and equal-physical-well-budget comparisons at
   5%, 10%, 25%, original fixed and random references, realized value and
   FDP/FPR together. Original purity requirements are not withdrawn.

Gram diagnostics cannot supply coordinate-wise intervals or uniquely identify
biological and technical variance components. They also do not establish
cross-compound campaign dependence or a sequential full-profile probe policy.
Compound bootstrap comparisons condition on the existing shared batches and
fitted rules; they are development uncertainty summaries, not certificates.

## Parallel execution

Shared geometry/utility identities and executable tests are checked first.
Reference sampling/evaluation and G training then run as separate workers with
bounded CPU threads. Both use the frozen common evaluator. No result from the
reference worker alters the G training protocol. Final comparisons wait for
both workers. The older paused experiments remain paused.

## Interpretation

Improved calibration alone is a distribution repair, not improved selection.
Improved ordering without calibrated risk is not authorization. Failure of this
one complete parametric geometry model does not prove all geometry predictors,
biological kernels, or probabilistic JEPA variants impossible. Route B and new
JEPA objectives are not included in this first comparison.
