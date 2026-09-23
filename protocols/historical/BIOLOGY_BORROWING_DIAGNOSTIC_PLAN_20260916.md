# Biological-reference intensity and incremental-value diagnostics

Recorded 2026-09-16, before execution of this round. This is a new development
experiment on already opened data, not a prospective certification or a claim
that its choices preceded earlier results. No world-model weights are trained.

## Questions and fixed core

1. Do legal target/MoA reference errors add predictive information beyond the
   existing amplitude-conditioned CORE when their contribution is not shrunk
   almost to zero?
2. Can decision-time support variables predict when borrowing helps, and does
   that prediction improve a held-out borrowing policy?

Retain STATE50 mean, original nine-coordinate geometry, scatter and amplitude
scaling, empirical radial family, original ADD_TWO Gamma/cost, lambda=.2,
stable-ID ties and the existing ten LINCS cell budgets (146 actions overall).
Target and MoA are evaluated separately. No new JEPA/SE/KAN network is trained.
Prior runs, protected FINAL and fifth repeats remain untouched.

## Populations and applicability

LINCS: the existing 1188 opened objects, five chemical-group folds and ten cells.
Each cell preserves MODEL_FIT, REF_FIT, DIST_CAL and QUERY membership. DIST_CAL
contains forty independent chemical-group representatives. QUERY outcomes never
fit a law, select alpha, fit a gate, set a cutoff or define a support subset.

LKCP: retain the prior MODULE_DEV 24h-reference to 48h-query construction.
Do not override unknown assay dose or condition mismatch to manufacture a BIO
comparison. When these checks fail, record NOT_APPLICABLE and exact CORE return;
do not call it an efficacy failure. No new resources or contexts are opened.

## Estimands

Primary mechanism population: supported QUERY objects for each relation, defined
solely from the actual allowed reference bank and pre-outcome metadata. Also
report the fixed union-supported population and the complete cohort. Do not
substitute the 36 previously active objects for all supported objects.

Primary mechanism scores are paired Gamma CRPS and NULL Brier, accompanied by
radial/full-vector log density to diagnose the information route. Report support
size, effective neighbor count, absolute strength, actual mixing, weight change,
five joint coverage levels and single/pair/average functional CRPS.

Decision estimand: the original full-query-cohort top-k plan at the original
budget, not a new budget spent only inside the supported subset. Report actual
value, NULL counts and replacements. Decompose policy-value changes on supported
and unsupported objects, since competition can change an unsupported object's
selection despite unchanged scores. These paired algorithm diagnostics are not
causal average treatment effects on treated subjects.

Chemistry-group paired resampling summarizes development uncertainty; layout
resampling is a dependence sensitivity. These intervals are conditional on the
fitted rules, not independent campaign certificates. Fixed-alpha curves are
descriptive multi-candidate diagnostics, not a license to select a query winner.

## Experiment A: fixed intensity

For relation R in {target, MoA}, use normalized legal relationship similarities
to form P_R from the same radial centers. Evaluate alpha in {0,.25,.5,1}:

    P_alpha = (1-alpha) P_CORE + alpha P_R.

Unsupported rows return P_CORE exactly. Remove the optional calibrated admission
and ESS shrinkage only for these named intensity ablations; retain all hard
context/identity/annotation restrictions. The target and MoA alpha=0 rows share
the same CORE. Keep the existing admitted-and-shrunk BIO as its own comparator.
An alpha=1 failure does not refute benefits at smaller alpha.

## Experiment B: incremental-value prediction

The target is delta_i(alpha,R) = log p_alpha(e_i) - log p_CORE(e_i), in the same
coordinates, with positive values denoting predictive gain. It is a noisy
realized proper-score difference, not an object's true noise law or Gamma gain.
Use the amplitude-conditioned CORE, never a weaker pooled law, as denominator.

Within each cell, fit new gates only using the forty permitted DIST_CAL groups.
The amplitude bandwidth remains the frozen REF_FIT estimate (independent of
CAL/QUERY), preserving the existing CORE recipe. Rebuild the empirical radial
law and its bandwidth inside every inner donor split. Never pool historical
outer-query OOF differences to create a seemingly independent gate dataset.

Three outer calibration folds evaluate the whole gate-building procedure;
inside their training parts, grouped CV produces held-out delta labels and
feature records. Donor identities and outcomes for a pseudoquery are excluded
from its density and support summaries. Finally refit the procedure using all
permitted DIST_CAL and predict the separate outer QUERY. This does not create
more than forty independent gate-development groups per cell. Report the much
smaller relation-supported counts and explicit constant fallbacks where needed.

Per relation, compare:

- CORE (never borrow);
- every prespecified fixed alpha from Experiment A (always borrow at that
  strength when legal support exists);
- FIXED_SELECTED: one alpha chosen from calibration-only held-out log scores,
  including zero, then applied uniformly to supported QUERY objects;
- DELTA_RIDGE: regularized linear incremental-value prediction;
- DELTA_BOOST: fixed-configuration nonlinear gradient-boosted incremental-value
  prediction; not a linear-only straw-man comparison.

Learned gates select among zero and the three preset positive strengths using
predicted expected delta, with deterministic ties preferring smaller alpha.
All algorithms are reported; QUERY results do not select which to deploy.
Features use only legal query X/amplitude, relation support, matching conditions
and donor records. Any normalization is fitted only on the relevant inner fit.
Save OOF delta, predictions, support, feature names, donor IDs and model settings.
R2 is diagnostic, never the sole admission/termination criterion.

## Evaluation implementation

Use 100000 complete-vector draws per endpoint distribution, common random
streams and the existing legal geometry-to-Gamma map. Because only mixture
weights change, CORE/reference predictive mixtures can be evaluated through
exact mixture identities: means and NULL probabilities are affine, log scores
use mixture density, and CRPS/energy include cross-component terms. Cross terms
must pair independent draw indices (exclude paired CRN diagonals where needed).
Joint radial CDF/quantiles use the actual mixed weights. This is an estimator
optimization of the same full distribution, not a reduced method or training
substitute. CORE replay and mixture endpoints are checked. Report differences
from earlier resampled-mixture Monte Carlo estimates rather than treating them
as model gains. If endpoint reuse is unavailable, use the original full sampler.

## Interpretation fixed before running

| Observation | Interpretation / next action |
|---|---|
| Fixed borrowing improves, learned gate does not beat fixed borrowing | Reference information may help, but individualized activation is not justified |
| Learned gate improves held-out proper scores over CORE and calibration-selected fixed borrowing | Evidence to pursue support-conditional borrowing, still requiring fresh validation |
| Radial NLL improves but Gamma/NULL do not | Measurement-distribution increment, not demonstrated acquisition increment |
| Gamma/NULL improve but original-budget value/risk do not | Task-distribution increment without allocation evidence |
| All tested strengths and gates lack resolved gain | No support for this relation/bank/interface in this development regime; not a universal biology impossibility |
| Full replacement worsens but intermediate strength improves | Shrinkage tradeoff; full-on/full-off alone would have been misleading |
| Gate fit has too few supported groups or unstable delta predictions | Limited support/power; do not reinterpret as knowledge being useless |
| Hard applicability fails | NOT_APPLICABLE, not a negative performance result |

Representation learning remains a separate next experiment. Its future baseline
must include a real nonlinear predictor of all legal descriptors, using the same
distribution family and evaluation, alongside amplitude-only and learned-state
models. This round's gate regressors do not constitute that representation test.
