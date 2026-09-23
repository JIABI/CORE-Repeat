# Mean and repeat-noise repair on opened source_5 DEV

## Question and scope

Does training-only moment anchoring improve the full conditional measurement
distribution and its derived acquisition predictions? This is a development
repair experiment, not a new kernel factorial or a certification run. Reuse
the completed seed 20260912 A checkpoint. No gradient retraining, model-size
change, JEPA, new biological prior, outcome redefinition, or threshold search.
The repair consists of fitting a complete statistical anchor and replacing
explicit output-distribution components. It is not a retrained OPAL backbone.

Use the unchanged 639-object, four-role, 3617-coordinate DEV export and its
383/96/64/96 train/validation/calibration/evaluation split. All new fitted
moments use the 383 training objects only. Keep all objects in all three
assessment partitions. No FINAL or fifth repeat is read. Old files are retained.

## Fixed variants

- **A_ORIGINAL**: existing fitted A; saved original results provide the comparator.
- **L_FULL_GAUSSIAN**: existing `ClosedFormBaseline`, fitted anew on TRAIN. Rank
  200, robust moment clipping at 8, noise shrinkage 0.05, positive full-coordinate
  residual floor 1e-6, random seed 20260912. Training-only affine standardization
  matches A. New conditioning inputs are not clipped. The residual outside the
  PCA span remains stochastic. This is not the truncated closed-form estimator
  previously reported at R2=0.218, and need not reproduce that value.
- **A_MEAN_ANCHOR**: replace only A's conditional mean with L's conditional mean;
  retain all of A's learned covariance components and environmental sharing.
- **A_MEAN_WITHIN_ANCHOR**: same mean replacement, plus replace A's diagonal and
  within-well low-rank covariance with L's fitted positive full-coordinate noise
  covariance. Retain A's learned compound and environmental shared covariance.

These comparisons isolate mean replacement and then within-well covariance
replacement. The latter changes total marginal variance: it is NOT an
equal-total-variance reallocation experiment. The anchor's residual covariance
includes unmodelled signal and batch interactions; it is not identified pure
instrument noise. Neither repair is presumed successful before measurement.

The inference adapters accept only one observed X and the already-legal
conditions/references. Future measurements enter scoring only. All variants
retain the original three action utilities, per-well cost 0.01, NULL <= 0 and
POSITIVE >= 0.005.

## Evaluation and reporting

First publish deterministic measurement-mean scores for all three partitions.
Then evaluate all three new variants with 2000 complete joint draws per object,
the original 5%, 10%, 25% budgets, and the existing original-space evaluation.
No subset, reduced feature space, or mean-only sampling replaces the full
distribution. Preserve within-object and retained cross-object shared draws.

Report mean MSE and R2, joint per-object NLL, coordinate and repeat-difference
coverage/width, full simulated-well cosine and RMS separately from mean-vector
geometry, Gamma bias/dispersion/ranking, NULL probabilities, and actual
same-budget value and risk. Retain poor scores as failure evidence. Unadjusted
paired compound intervals describe these fixed fits on historical DEV, not
independent certification or new-source uncertainty.

Do not select a variant or change a contract based on a single favourable
partition. The first result answers whether the parameter repair helps and
which component matters. Further neural fitting or kernel experiments are
separate decisions after these diagnostics.

## Resource handling

The existing second-seed factorial process is suspended in memory while this
bounded repair runs, retaining its checkpoint and queue. A runner records its
process identity and resumes that same process on completion or exception.
No old experiment configuration or result is overwritten. Progress is kept
locally; notify the user with an actual completed result, not every few epochs.
