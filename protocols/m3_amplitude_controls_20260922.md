# M3: amplitude and dispersion allocation controls

Specified on 22 September 2026 before fitting the new controls. The user
authorized the complete four-resource comparison and the post-hoc confirmation
controls. The experiment asks how much fixed-budget repeat value is recovered
by simple first-well amplitude rules, a fitted nonlinear amplitude-only policy,
and a policy prioritizing predicted repeat dispersion.

## Development populations and access

Reuse the original 60 R2 deployment cells: EU 904 objects/5 cells, JUMP
639/5, LINCS 1,188/10 and RxRx3 10,410 compound-dose conditions/40. Use the
exact C3 role loader and original object identities, chemical groups, layouts,
TRAIN, REF_FIT, VALIDATION, DIST_CAL, DEV_EVAL and fixed quotas. TRAIN plus
REF_FIT supplies fitted outcomes; VALIDATION selects hyperparameters;
DIST_CAL supplies only the probability calibration. DEV_EVAL outcomes do not
choose an arm, direction, function or hyperparameter. Original CORE and
calibrated HistGB lists and predictions are reused unchanged by identity.

## Controls and complete configurations

Amplitude is the log root-mean-square norm of X in the original prepared
measurement space: a = 0.5 log(mean_j X_j^2). Within each cell, raw ascending
and descending amplitude are separate parameter-free rules. Each takes the
original k objects, breaking ties by ascending object identity.

The nonlinear amplitude-only policy fits a HistGradientBoostingRegressor to
Gamma and a HistGradientBoostingClassifier to 1[Gamma <= 0], using a as the
sole input. Each uses 200 iterations, learning rate 0.05, L2=1, max_bins=255,
no early stopping, and the saved cell seed. Candidate (max_leaf_nodes,
min_samples_leaf) settings are (7,20), (15,10), (31,10). Select the regression
by VALIDATION MSE and the classifier by VALIDATION binary log loss. Ties
within 1e-12 retain the earlier, simpler setting. Retain the TRAIN+REF_FIT
fit, without refitting on VALIDATION. Clip predicted Gamma to [-1.02,0.98].
Fit the existing fixed-L2 Platt calibration on DIST_CAL raw classifier
probabilities only (C=1, saved implementation). Rank descending by the
separately predicted E[Gamma] - 0.2 P(NULL), with ascending-ID ties. This is a
scalar decision predictor, not a coherent Gamma law or joint well model.

The predicted-dispersion control fits the same HistGB regression grid to
log W, where W = sum_(r=1..3)||Y_r - mean(Y)||^2/(2D), with Y=(Z1,Z2,V).
It uses the full original ACCESS_MATCHED first-well representation and
chemical features, unlike the amplitude-only arm. Select by VALIDATION MSE,
retain the TRAIN+REF_FIT fit, and rank by predicted log W DESCENDING. This
fixed direction prioritizes larger predicted repeat variation and will not
be reversed after evaluation. It is not the Gamma-minus-risk score. No
query repeat measurement is an input to either learned control.

Implementation: scikit-learn 1.9.1, with the existing C3 role loader and
original prepared data. All specified fits and all 60 cells are included;
there are no reduced-data or shortened-training runs. Original full CORE
and HistGB methods remain the historical comparators.

## Evaluation and saved outputs

Freeze every new query score and selection before reporting its outcome.
Report total/selected-mean/per-candidate Gamma, observed NULL counts/rates,
the cell-quota-matched exact random expectation, overlap with each policy,
and per-cell results. Report learned amplitude mean MSE, NULL Brier and
forecast selected NULL; report dispersion log-W MSE. Save per-object inputs,
predictions, realized endpoints, original/new selections, fitted model
checkpoints, chosen settings and validation losses once for reuse.

Use 2,000 paired fixed-list block resamples for chemical groups and for
saved layouts, seed 20260922. Repeated doses share chemical-group weights.
Report new-arm minus CORE, HistGB and exact random value and NULL
contributions per candidate. Models and lists remain fixed; these intervals
describe dependence sensitivity conditional on the fitted models. Also
report amplitude-only minus each amplitude heuristic and dispersion control.
Report cluster counts. Do not interpret unresolved intervals as equivalence.

Record fit/prediction time, peak process memory, action wells and fitting
role counts per deployment cell. Parameter-free controls need no fitted
reference/calibration pool. Learned amplitude uses TRAIN+REF_FIT,
VALIDATION and DIST_CAL; dispersion uses TRAIN+REF_FIT and VALIDATION.
Physical fitting wells are four times the relevant distinct fitting-role
objects. Do not sum cross-validation acquisitions as one deployment cost.

## Post-hoc confirmation controls

Use the saved 1,539 qualified identities, 1,527 eligible first wells and
k=192. Preserve original R4 artifacts and frozen CORE/HistGB lists. Recompute
both parameter-free amplitude rankings by saved norm2_per_feature and stable
ID. Also fit the same amplitude-only configuration using the existing final
DEV904 TRAIN434+REF_FIT181, VALIDATION108, DIST_CAL181 split; it uses no
confirmation outcome. This new model and both heuristic lists are post hoc,
not prespecified confirmation policies. Freeze them in the new M3 directory.

For Gamma, use the known range [-1.02,0.98], report observed sums together
with missing-outcome identification bounds, and use inclusion probability
192/1527 for the random reference. NULL bounds assign each missing Gamma
both possible labels. For differences, use signed selection weights so
shared missing objects cancel. These bounds are not confidence intervals.

The paired external endpoint is the mean MEDINA/USC neighbourhood-Spearman
improvement, observed only when both sites are observed, with range [-2,2].
For every rule and the random reference use the same eligible population
without conditioning on Gamma availability. Report observed counts and
available-case means, fixed-list mean bounds, site-specific availability,
and paired bounds on rule contrasts. Gamma and NULL share missingness;
cross-site missingness is reported separately. Use 2,000 chemical/layout
fixed-list block resamples of the bound endpoints for descriptive sensitivity.

## Execution

Run at most two CPU workers with one numerical thread each; no GPU. Complete
cells are checkpoints and are not fitted again. Freeze this protocol before
the first fit. Any implementation correction is recorded separately without
altering the scientific configuration in response to outcomes.
