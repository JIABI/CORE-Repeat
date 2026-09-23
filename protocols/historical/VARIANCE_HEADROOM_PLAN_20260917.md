# Sampling precision and bounded variance-adjustment headroom

This is a diagnostic on the already-opened 1,188 LINCS Cell Painting objects. No new measurement population, model training, original selection change, or main-method adoption is part of this round.

## Questions

1. Are the frozen CORE, GELU and structured-adapter scores numerically stable enough to resolve their observed differences and per-cell selection boundaries?
2. Within the existing pair-sensitive/remainder variance-adjustment family, how much can hindsight fitting improve the original Gamma CRPS?
3. How much of that improvement is retained by a scalar chosen on calibration objects, or by the existing decision-time error-energy prediction passed through the same calibration mapping?

The oracle is an outcome-dependent diagnostic, not a deployable predictor, a conditional-information ceiling, or evidence of prospective calibration.

## Fixed scope

Keep the saved CORE mean, empirical amplitude-conditioned radial law, rank-three/rank-six geometric directions, original Gamma and .02 action cost. Keep the five chemical-group folds, ten query/calibration cells, cell budgets and score E[Gamma]-.2 P(NULL). Use the same legal-support mask as the dual-branch experiment for primary contrasts. Unsupported objects return CORE. Report supported and full-cohort scores separately.

The two total log-variance increments range over [-log(4),log(4)], matching the combined two-branch range. This is a geometric scatter adjustment, not an identified physical noise decomposition. The main finite candidate set is a 9-by-9 grid, with zero and the scalar diagonal included. No claim of a continuous global optimum is made.

## Precision diagnostic

Compare CORE, GELU and DUAL_STRUCTURED without changing their saved distributions. Use two fresh independent seeds, 100,000 draws per object per seed and nested 10,000/20,000 prefixes. Draws are shared across arms. Compute each draw's score Gamma-.2 I(Gamma<=0), its Monte Carlo standard error and the covariance of Gamma and the indicator. Use 20 independent Monte Carlo blocks for paired CRPS integration-precision diagnostics; keep biological resampling uncertainty separate.

For selection, each cell retains its original integer budget. Boundary refinement uses predicted score and Monte Carlo uncertainty only, never realized Gamma. A four-SE overlap rule identifies candidates; the bounded precision diagnostic can increase those objects to 500,000 then 1,000,000 draws. Report unresolved cutoffs at the limit rather than claiming exact rankings. Original selected lists remain unchanged in their source files.

## Headroom comparisons

- CORE: no adjustment.
- GLOBAL_CAL: one common scalar chosen using current-cell calibration outcomes, with group-LOO calibration radial laws and zero among the candidates.
- LEARNED_RANK_CAL: the existing decision-time error-energy score, with a monotone scalar calibration mapping. No new feature learner is fitted.
- AMPLITUDE_RANK_CAL: the saved amplitude-only predictor, passed through the identical mapping, to isolate information beyond amplitude.
- TRUE_ENERGY_RANK_CAL: the corresponding realized error-energy rank, passed through the same form of calibration mapping. Its query input uses future outcomes; label it hindsight explicitly.
- H1_CRPS: per-object scalar chosen to minimize Gamma CRPS on optimization draws.
- H2_CRPS: per-object two-component grid choice minimizing Gamma CRPS.
- H1_NLL and H2_NLL: analogous choices minimizing complete-vector NLL. These test objective mismatch; they are not Gamma-CRPS bounds.

Optimization uses 4,096 Monte Carlo draws and common random numbers over candidates. Final scoring uses independent 100,000-draw streams, paired across arms. Retain the candidate search values and the final independent-MC values separately. If a small apparent headroom would drive a stopping decision, first assess grid and optimization-MC sensitivity; an incomplete numerical search cannot establish a continuous-family upper bound.

The calibration map is fitted only on the 40 disjoint current-cell representatives, with leave-chemical-group-out radial laws. It must not use a representative's covariance cached in its opposite query role. The existing energy predictor may be read on calibration and query objects only if both are excluded from its fitting groups. If this cannot be verified, omit the deployable rank arm and state the missing evidence rather than using query outcomes to fit it.

The rank maps use mid-ranks of the calibration predictor and its empirical CDF for query scores, followed by increasing bounded isotonic regression to the calibration-only CRPS-optimal scalar. The historical .390/.445 comparison refers to rank-six geometric error energy whitened by the original RIDGE OOF covariance. Use that same energy definition for its true-rank comparator; retain current-CORE total energies separately. Rank-three/rank-six are Jacobian-defined geometric blocks, not norm/angle coordinates. Current Gram geometry already removes the common first-hole scale.

## Interpretation

Primary score: Gamma CRPS. Co-report NULL Brier, joint NLL, joint-region coverage, predicted Gamma, NULL discrimination and same-budget realized value. Coverage and selection for hindsight arms are outcome-contaminated diagnostics, not performance certificates. Optimizing CRPS does not bound Brier or decision value.

Use 0.0001 absolute Gamma-CRPS reduction as a declared small-effect diagnostic threshold and 0.0005 as substantial headroom for this round; these are practical study choices, not universal information limits. Report paired chemistry-group and layout intervals. A point estimate below twice its interval half-width is not an equivalence test. A small estimated hindsight gain supports lower priority only for this fixed adjustment family, and only after numerical-search uncertainty is checked. A large gain proves expression headroom, not learnability from the first hole.

The 0.149%/0.051% mean-MSE improvements from the previous direction diagnostic apply only to its fitted single-global-coefficient family on those observed objects; do not relabel them as bounds for all biological models.

No extra SE gate, JEPA, larger network, threshold sweep on protected data, or main-model replacement is started by this diagnostic.
