# Conditional joint prediction error: frozen-mean LINCS experiment

## Question

Can first-well morphology direction, magnitude and existing reference outcomes improve the joint predictive distribution beyond STATE50? Separate overall scale, coordinate-specific scale, globally repaired dependence, and locally conditioned dependence. This is a complete statistical reference-memory experiment, not a reduced substitute for a future JEPA model.

## Scope and information

Use the existing 1,188 LINCS Cell Painting objects, four roles, original 1,694-feature measurements and the five frozen STATE50 models from `lincs_state_biology_20260916_v1`. Preserve the original endpoint, preprocessing, costs, previous results and protected FINAL/fifth-repeat data. The new branch changes no mean weights. Target/MoA, cell counts and JEPA are excluded from this first stage.

Reuse the exact ten donor/query cells saved in `lincs_reference_information_20260916_v2`: each original outer test fold is split by chemistry group, then swapped. Neither donor nor query outcomes trained its frozen base model. Each query appears once. Tune covariance parameters only through donor leave-chemistry-group-out scores. Rank within query cells, never across swapped cells; keep the existing allocation of 146 selected objects / 292 additional query wells. Reference acquisition costs remain excluded, so this is an archival auxiliary-reference diagnostic, not prospective cost certification.

## Retrieval

Generic similarity is the mean of positive first-well cosine and binary Morgan Tanimoto. Average this with an RBF similarity of log first-well norm; its bandwidth is `max(std(log norm of original FIT first wells), 0.1)`. Use the same fixed bandwidth for donor-LOO and query predictions. Select the top 16 eligible references, deterministic identity tie breaks, no own chemistry group. Unsupported similarity falls back to all eligible donors. No outcomes determine the keys or bandwidth.

## Error model and six arms

Work in the existing fold-standardized nine-dimensional log-Cholesky geometry. For each reference, r is actual geometry minus frozen STATE50 mean. Use M_i=sum_j w_ij r_j r_j^T, the second moment around that frozen mean. Do not discard residual bias by centering. These are prediction-error moments, not identified biological/technical noise variances.

Let C0 be the inherited positive-definite covariance, v0 its diagonal and R0 its correlation matrix. All models are Gaussian in the same nine-dimensional coordinates and use the same frozen mean.

1. BASE: C0 unchanged.
2. GLOBAL_SCALE: uniform eligible reference second moments. tau=tr(C0^-1 M)/9; C=(1-beta+beta*tau) C0.
3. LOCAL_SCALE: identical scalar operation, using direction+magnitude reference weights.
4. LOCAL_DIAG: v=(1-beta)v0+beta*diag(M_local); retain R0 and assemble C=diag(sqrt(v)) R0 diag(sqrt(v)).
5. LOCAL_GLOBAL_CORR: keep exactly the LOCAL_DIAG variances and selected beta. Repair correlation using uniform eligible reference M_global: Rm=corr(0.25*C0+0.75*M_global), R=(1-eta)R0+eta*Rm.
6. LOCAL_JOINT: same variances and beta again, but use M_local for the correlation repair.

Select beta and eta from {0,0.25,0.5,0.75} by full donor-LOO Gaussian NLL; ties favor smaller changes. The joint arms first reuse LOCAL_DIAG beta and then select eta. Thus correlation comparisons have matched marginal variances. All covariance candidates retain positive-definite baseline support; no unreported eigenvalue clipping. Zero strength recovers the baseline exactly.

## Evaluation fixed before scoring

- Full Gaussian NLL and energy score in fold-standardized nine-dimensional geometry; coordinate and joint ellipsoid 95% coverage. NLL is not sufficient for a downstream win.
- 10,000 joint draws per query, common random numbers across arms. Map through the original legal geometry factor without clipping, resampling or modifying Gamma.
- Single future-well relative squared norms (three roles), all three future-pair squared differences, three future-pair average squared norms and the three-future-well average squared norm: log1p transform for CRPS, central 95% coverage and interval width. These are observable geometries, not physical variance estimates. X is observed and held fixed, not resampled.
- Also score absolute future-pair spread in the original normalized/clipped feature units, multiplying relative squared differences by observed ||X||^2 / feature count before log1p. This checks whether a gain is only about the known X denominator.
- Original ADD_TWO Gamma fair CRPS, mean error, Spearman, NULL Brier and central 95% coverage; same-budget realized net value and NULL counts.
- Save per-object predictions, full covariance matrices, role-specific scores, choices and support. All means must be identical across arms. LOCAL_DIAG/LOCAL_GLOBAL_CORR/LOCAL_JOINT marginal variances must agree exactly up to roundoff.

Primary mechanism comparisons are LOCAL_SCALE minus GLOBAL_SCALE, LOCAL_DIAG minus LOCAL_SCALE, LOCAL_JOINT minus LOCAL_DIAG, and LOCAL_JOINT minus LOCAL_GLOBAL_CORR. Also compare each arm with BASE. Report chemistry-group paired bootstrap intervals and layout-cluster sensitivity, conditional on saved models and donor choices. Repeated development, model refitting and reference resampling uncertainty are not covered. Fold results and first-well amplitude strata are descriptive; they do not alter selection or remove objects. No outcome-based repartitioning.

## Interpretation and subsequent stages

An improvement in joint geometry alone is not a claim of better allocation. Report separately distributional information, Gamma/NULL prediction, and policy value. If local dependence has no stable incremental value, do not call the experiment proof that dependence is unimportant in biology. If it does improve, target/MoA can next be tested as an independent retrieval increment. JEPA remains a later matched representation comparison, not included in this run. No certification thresholds are relaxed.
