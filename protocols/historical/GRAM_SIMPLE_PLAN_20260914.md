# Same-target geometry baselines and neighbor support

## Question

Does a low-complexity conditional model extract individual acquisition-value
information more reliably than G_DIRECT, and are there enough decision-time
neighbors to motivate a later local-borrowing experiment?

This is a development follow-up to `gram_probability_20260914_v1`. The original
639 opened compounds, TRAIN 383 / validation 96 / calibration 64 / evaluation 96,
four roles and 3,617 profile coordinates remain unchanged. No original FINAL,
fifth repeat or additional candidate is used. Original Gamma, action costs and
the seven-item contract are unchanged. These reused DEV partitions are not a
new independent validation or certification cohort.

## Models

All new predictions use the existing nine standardized log-Cholesky coordinates
u of G = WW^T / ||X||^2. The observed G00=1 and full positive-definite Schur
complement are preserved by the same decoder. Original utility is computed for
each joint sample. Norms are retained; no endpoint whitening, truncation,
clipping, target averaging or outcome-dependent exclusion is introduced.

* GLOBAL_GEOMETRY: TRAIN mean and Ledoit-Wolf full joint covariance in u. It
  does not condition on X. Identical Monte Carlo draws are broadcast to every
  object, so population-level predictions have exact ties. This common-random-
  number device evaluates marginal laws; it is not a model of shared campaign
  noise. Selection performance is the uniform-subset expectation, not the
  outcome of an arbitrary lexical-ID tie break.
* RIDGE_GEOMETRY: complete G input, consisting of all 3,617 scaled X coordinates
  plus the standardized log raw X norm. A multi-output ridge predicts u with an
  unpenalized intercept. Candidate penalties are [0.01, 0.1, 1, 10, 100] for
  mean squared residual summed over outputs plus lambda times squared coefficient
  norm; the corresponding unnormalized ridge penalty is n*lambda. Five-fold
  TRAIN CV chooses the minimum u-coordinate MSE, numerical ties (rtol 1e-10,
  atol 1e-12) preferring larger lambda. There is no selection using held-out
  Gamma or purity.
* RIDGE uncertainty uses a separate TRAIN outer-five/inner-four-fold procedure:
  each outer fold selects its penalty without that fold's target rows, then
  predicts those rows. The resulting residual second moment is estimated as
  Ledoit-Wolf covariance of centered residuals plus residual-bias outer product.
  The bias is NOT also added to the mean. This is predictive-error dispersion,
  not an identified decomposition of biological and technical noise.
* All internal folds share the already fixed outer-TRAIN G input and target
  preprocessing, and fit their own intercept/centering. Thus this is nested
  penalty selection conditional on fixed preprocessing, not a fully cross-fitted
  preprocessing pipeline or an independent estimate of full-sample risk. OOF
  residuals also include parameter uncertainty from smaller fitted samples.
* Existing L_GRAM and best-epoch-10 G_DIRECT predictions are reused unchanged.
  L generates full spectra before mapping to geometry. G has nonlinear means
  and heteroscedastic covariance; RIDGE changes both capacity and covariance
  specification. This is a complete baseline comparison, not a single-factor
  architectural ablation.

## Evaluation fixed before fitting

Seed 20260914; 2,000 joint samples per object; the same existing geometry and
policy evaluator, TRAIN metric scales, 2,000 compound bootstrap draws and 2,000
matched-random subsets. No new model is selected using validation/calibration/
evaluation results. New model fits and internal CV decisions are saved.

Report nine-coordinate prediction error (including TRAIN OOF), joint geometry
energy score, cosine/norm/difference/average predictive distributions, original
Gamma CRPS, NULL Brier/AUC, rank correlations, and all existing 5/10/25% policy
budgets. For the common physical-well cap floor(q*N), ADD_TWO activates
floor(floor(q*N)/2) compounds. Report realized value, FDP and FPR jointly.

For identical-action/budget comparisons, use paired frozen selection-mask
differences; GLOBAL's mask is its uniform selection probability k/N. Report
compound bootstrap intervals conditional on the fitted predictions and these
shared batches. No model-selection uncertainty or new-batch guarantee is
claimed. Constant predictions have undefined correlation and AUC 0.5 when both
classes exist, not random Monte Carlo ranking skill.

The interpretation is conditional: a reliable conditional-over-global gain is
evidence that this predictor uses decision-time individual information. A null
comparison is not an information-theoretic impossibility result. Probability
score improvement without value/risk improvement remains a distribution result.

## Parallel neighbor-support audit, not a kernel experiment

Every query may borrow only from the original 383 TRAIN compounds. TRAIN
queries exclude their own ID. Inputs are X and the existing chemical
fingerprints; Z1/Z2/V, actual Gamma and future-derived QC never select neighbors.

Morphology: all-coordinate TRAIN-scaled cosine distance and RMS Euclidean
distance, separately. Chemistry: Tanimoto on the existing Morgan radius-2,
512-bit fingerprint, excluding the appended valid-SMILES indicator. Validate
fingerprint and metadata coverage; unknown mechanism annotations remain unknown.

Report top-one/five/ten neighbor distances or similarities, chemical counts at
Tanimoto >=0.3/0.5/0.7, and top-20 normalized exp(-distance/h) effective-neighbor
number, with h the median TRAIN leave-self-out fifth-neighbor distance. These
thresholds are descriptive, not declared proof of useful biological similarity.
Read effective-neighbor counts jointly with absolute distance: twenty equally
distant, unrelated objects are not twenty good neighbors. Report query support
relative to TRAIN leave-self-out distances, morphology/chemistry top-ten overlap
and relevant available experimental-condition metadata.

The audit does not test whether neighbors share response or noise distributions.
No kernel, JEPA, neural continuation or new data acquisition is run this round.
