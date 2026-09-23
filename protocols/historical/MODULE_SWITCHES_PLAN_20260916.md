# Optional biology and representation switches

2026-09-16. This is a new developmental comparison authorized by the user. The
STATE50 mean, nine-coordinate geometry, original ADD_TWO Gamma, cost and frozen
score E[Gamma] - 0.2 P(NULL) remain unchanged. Existing runs are not overwritten.

## Questions and scope

Biology: can target/MoA relationships select more appropriate measured reference
errors beyond amplitude? Representation: can an X-only state embedding select
more appropriate measured reference errors beyond amplitude? Both initially act
on the conditional empirical radial reference weights. They do not claim to
identify physical shared/independent noise or repair radial-direction dependence.
Changing radial weights may change covariance as well as tails.

LINCS uses the already opened 1188 objects and existing five-fold, ten-cell roles.
All new comparisons are development evidence. LKCP Batch2 is assigned MODULE_DEV
for resource/compatibility assessment and any measurements opened in this round.
It is not assigned PRIMARY_EVAL. Contexts remain distinct. No old protected FINAL,
old fifth repeat or qualification/custody state is used. A missing compatible
well-level resource is reported as unavailable, never replaced with consensus.

## Operational switches

Each switch distinguishes resource eligibility, calibration-selected admission,
and per-query support. Eligibility is not a demonstrated performance gain.
All gates use X, declared metadata and independent reference records; query
future measurements are used only after predictions are fixed for evaluation.

* BIO: independent target and MoA radial-retrieval channels. Their coefficients
  are selected from (0,0), (.25,0), (.5,0), (0,.25), (0,.5), (.25,.25).
  Chemical Tanimoto >= .7 is not a prerequisite. Incompatible cell line, time or
  dose receives zero relation support; unavailable matching fields are unknown
  and do not silently count as matching. Unknown mechanism is not absence.
* PCA_STATE: train-only linear state baseline for nonlinear representations.
* DIRECT_STATE: small ordinary nonlinear state encoder.
* CONDITIONAL_STATE: role-conditioned anchored latent predictive encoder,
  JEPA-like in predicting a fixed future representation, not the old JEPA
  objective and not an EMA-teacher implementation. Raw amplitude is retained.

Representations train only on MODEL_FIT, with an internal group-heldout split for
preprocessing, target PCA and early stopping. They never fit on DIST_CAL/QUERY
future wells. Hyperparameters are fixed in the representation implementation.
No universal N or annotation-coverage cutoff is asserted.

Per-channel relevance uses feature-only normalized weights and continuous
shrinkage from effective neighbor count and absolute similarity. Independent
target/MoA coefficients avoid forced shared coefficients. Off/unsupported rows
return the exact core weights, not a different generic postprocessor.

## Selecting a switch without query outcomes

The existing DIST_CAL radial reference set is partitioned by chemical group.
For each internal fold, rebuild the radial law, including bandwidth, from only
the remaining calibration references. Evaluate full-vector radial log density
on heldout references. Choose the smallest nonzero mixing coefficient only if
its mean loss beats the zero coefficient by more than one group-level standard
error of the paired difference; otherwise use zero. This is a conservative
developmental selection rule, not a hypothesis test or deployment certificate.
Refit the radial law on all DIST_CAL after selecting coefficients. Query support
is recomputed against that actual bank, not an earlier bank's coverage counts.

## Comparisons and evaluation

Arms: unchanged CORE, CORE+BIO, CORE+PCA_STATE, CORE+DIRECT_STATE,
CORE+CONDITIONAL_STATE. Modules are tested separately, not factorially combined.
Sampling uses the same streams and 100,000 joint draws per object. Exact off
replay is tested separately from Monte Carlo error. Supported competitors can
change an unsupported object's top-k membership without changing its score.

Report resource eligibility, calibrated admission, per-query support, realized
nonzero contributions, heldout radial NLL/energy, five coverage levels,
single/pair/average functionals, Gamma CRPS, NULL Brier/AUC, and actual same-budget
value and NULL counts. The old ten-cell budgets are retained for LINCS; ties use
stable IDs and lambda=.2 throughout. Include lambda=0 and same-budget uniform
random descriptive references. Chemistry/layout resampling measures sensitivity
conditional on fitted predictions; it is not a new finite-sample certificate.
No outer query performance selects a winning gate or rewrites the baseline.

Primary scientific readout: heldout Gamma CRPS and NULL Brier, with joint radial
NLL diagnosing the retrieval mechanism and selected value/risk reported together.
A radial-only gain does not establish improved acquisition. Few/no supported
queries is a coverage limitation, not proof that biological knowledge is useless.

## LKCP execution boundary

Before result comparisons: establish individual-well resource access, matching
feature definitions, actual treatment conditions, donor identity isolation and
compatible preprocessing. A model trained for Pilot1 coordinates cannot be
silently applied to unmatched columns/normalization. If compatible data require
a new core fit, retain the frozen training recipe and fit it on declared LKCP
development partitions before comparing optional modules. Resource failure is
not a negative model result. No paid data or large unbounded download is authorized.
