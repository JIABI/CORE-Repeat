# OPAL 2: training-objective comparison

Prepared 11 September 2026, before fitting the three comparison arms.

## Question

Does directly optimizing the deployed predictive distribution, followed by an
explicit score for its derived acquisition utility, improve out-of-training
utility prediction and allocation? The previous 38-training/10-evaluation run
motivates this experiment; its negative correlation is not a population claim.

## Data and unchanged endpoint

Use only `data/source5_primary_fullcontrols`: the 639 previously opened JUMP
source_5 DEV compounds, their four already available roles, all 3,617 outcome
coordinates, and the existing DMSO controls. Retain the saved compound split:
383 training, 96 validation, 64 calibration, 96 evaluation. The old study FINAL,
fifth repetitions, other candidates, spatial outcomes, and other sources are
outside this experiment. These are development comparisons, not new independent
certification evidence. The 10 previously inspected evaluation compounds remain
in the evaluation partition; they are not represented as newly blinded data.

The state is X; acquired roles are Z1 and/or Z2; V is verification only. For an
action with k extra wells, utility is the original fixed-space half-cosine
improvement minus 0.01k. NULL remains utility <= 0; POSITIVE remains utility >=
0.005. STOP has zero incremental utility. No endpoint, missingness rule, or
original seven-item contract is changed. Existing reference controls are sunk
resources in this comparison; a newly purchased reference panel is not simulated.

## Three arms

| Arm | Probability-training objective | Purpose |
| --- | --- | --- |
| A_ELBO | Existing conditional ELBO + existing auxiliary terms | Matched original training control |
| B_NLL | Predictive joint NLL + the same auxiliary terms | Remove the target-posterior variational approximation |
| C_NLL_CRPS | B + derived-utility CRPS, weight 1.0 | Explicitly score the downstream utility distribution |

The Gaussian joint likelihood is exact within each minibatch. Minibatches do
not constitute evaluation of a single full-campaign joint likelihood.

C uses the same training compounds as its likelihood minibatch, with fixed
X -> (Z1, Z2, V) auxiliary tasks. Each Monte Carlo draw jointly samples all three
future wells. Profiles are returned to the unchanged outcome coordinates before
computing the three non-STOP utilities. The CRPS uses the off-diagonal unbiased
sample statistic, 16 draws, and equal action weights. Its coefficient is fixed
at 1.0 without an evaluation-driven search. Training uses differentiable cosine
utilities, not hard NULL/POSITIVE indicators. Calibration and evaluation targets
never enter the gradient.

## Common model and fitting procedure

Retain the full conditional model: 256 hidden units, two grouped-attention
layers, four heads, compound/environment rank 32, residual rank 8, measurement
law/KAN operator, chemistry, actual available references, training-only library
context, and the complete low-rank-plus-diagonal joint Gaussian decoder. No
feature truncation or architecture reduction is introduced.

Use seed 20260911. Fit the full conditional JEPA stage for 60 epochs on the 383
training compounds once, then reuse that identical fitted encoder in all arms.
All arms start their other parameters identically. Role sampling, minibatch
order, and keyed main-training randomness are paired across arms. C's auxiliary
randomness has a separate stream. Sharing the JEPA artifact is an exact reuse
of common work, not reuse of the earlier 38-compound encoder.

Probability training: at most 100 epochs, batch size 32, AdamW learning rate
0.0003, weight decay 0.0001, gradient clipping 5.0, validation patience 15,
minimum improvement 0.0001, and the existing learning-rate scheduler. All arms
select checkpoints by the same validation predictive NLL. This isolates the
training objective; task-based checkpoint selection is not introduced at the
same time. Maximum schedule is matched; actual early-stopping epochs may differ.

Use the existing deterministic CPU implementation. Do not silently move the
float64 likelihood to an unsupported accelerator or reduce the model to shorten
the run. Save checkpoints, configuration, fitting identities, training history,
and the source snapshot used for this comparison. Existing runs are preserved.

Two common numerical identities apply to every arm: an empty context returns
the exact prior, and likelihood integration omits environmental latent columns
that affect only unobserved padding. The latter integrate to one. Actual
measurements, observed unknown-environment factors, covariance ranks, and future
joint sampling are unchanged. Value/gradient equivalence is checked before use.

## Evaluation and interpretation

Only after all three models have finished fitting, evaluate their frozen best
checkpoints with 2,000 joint Monte Carlo draws. Report every arm; no winner is
chosen to change this run's settings.

1. For each action: Pearson and Spearman correlation, R-squared, POSITIVE AUC,
   NULL/POSITIVE Brier scores, predictive utility CRPS, and interval coverage.
2. Fixed ADD_TWO selection at 5%, 10%, and 25% coverage, separately ranked by
   predicted mean utility and POSITIVE probability. Keep the same number of
   acquisitions when comparing with uniform random selection.
3. The existing multi-action planner, fixed strategies, and action-mix-matched
   random policies. Do not compare different expenditures as if budgets match.
4. Fixed-coordinate predictive NLL and nominal 90% coordinate coverage. Also
   report finite-sample conformal diagnostics without calling them certificates.
5. Existing direct-head and conditional-residual baselines, with their actual
   input definitions retained.

Interpret B-A as the effect of the likelihood-training change in this setting;
interpret C-B as the incremental effect of the declared utility score. An
improvement in NLL or CRPS alone does not establish better allocation. Positive
point estimates alone do not establish a reliable advantage. Paired compound
resampling, if reported, is conditional on the observed batch layout and does
not solve shared-batch dependence. One training seed cannot establish robustness
to retraining or performance at unseen sites.

Report no improvement, mixed results, or reversal as observed. Do not invert
scores, sweep thresholds, change CRPS weights, drop evaluation objects, or open
FINAL in response to these results.
