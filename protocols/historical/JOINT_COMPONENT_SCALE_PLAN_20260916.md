# Conditional contrast-sensitive error scales

## Question and scope

On the same 1,188 opened LINCS objects, test whether a separately fitted amplitude-dependent scale for pair-difference-sensitive error improves on one amplitude-dependent scale for all error directions. Keep STATE50 means, original Gamma, ten reference/query cells, 146 selections, and old outputs unchanged. No target/MoA, cell-count addition, JEPA, FINAL, or new objects.

The physical model Y_ir=mu_i+b_i+epsilon_ir would imply Var(difference)=2e and Var(mean)=s+e/k only after specifying conditional means, exchangeability, and error independence. Raw replicate averages also contain phenotype. The present geometry model does not identify physical s/e. Original-profile moment and amplitude-regression diagnostics are reported separately, with the common-energy quantity named a proxy, not technical shared variance.

## Adapted predictive-error branch

For each frozen standardized geometry mean m and existing LOCAL_FIT covariance C=LL^T, calculate the exact derivative J of the three log(1+future-pair squared-distance) observables with respect to standardized geometry. Let P project onto the row space of JL. It has rank three; verify this for every object. The complementary six-dimensional subspace is locally invisible to those three observables, not necessarily physical shared noise.

Whiten reference errors w=L^-1(y-m). The fitting targets are ||Pw||^2/3 and ||(I-P)w||^2/6. Reference covariances and weights exclude the entire object's chemical group. STATE50 itself was fitted outside the whole outer fold. Fit zero-mean predictive-error scales; do not remove residual bias or use query outcomes.

The new covariance is L[a_D(x)P+a_R(x)(I-P)]L^T. It is positive definite for positive scales; both total variance and the allocation of error directions may vary by object. This is a full nine-dimensional Gaussian conditional law, with unchanged legal Gram decoder. It is an operational contrast-sensitive decomposition, not an identified physical variance-components model.

## Four fixed arms

1. LOCAL_FIT: unchanged baseline, both multipliers exactly one.
2. SPLIT_CONSTANT: two intercept-only multipliers fitted from reference errors.
3. AMPLITUDE_TOTAL: one log-linear multiplier of observed log||X||, applied to all nine directions.
4. AMPLITUDE_SPLIT: separate log-linear multipliers of the same input for contrast and remainder directions.

Scale fitting minimizes Gaussian projected negative log likelihood, averaged over fitting reference objects. The total arm has slope penalty 0.5*theta_slope^2 (lambda=1). Split penalties are weighted by subspace dimension: lambda_D=3/9 and lambda_R=6/9, so equal split slopes have exactly the same combined penalty as a single common slope. Intercepts are unpenalized. Input standardization uses fitting references only. No hyperparameter search or query-based selection. Calibration objects remain unused for fitting or selection in this comparison. These arms isolate splitting vs amplitude conditioning, conditional on the existing morphology/chemistry/amplitude retrieval covariance.

Reuse the previous 78–81 fitting-reference objects, 40 calibration chemistry groups, and query cells. Fit all four arms on the same reference pool. The existing donor leave-group-out choice is retained, not advertised as independently validated within that pool. Report scales, coefficients, ranks, optimizer convergence, and complete per-object predictions.

## Evaluation

Use 10,000 joint draws per object, shared random normals across arms, unchanged Gamma and selection budgets. Report original-space-derived single, difference, and average scores; NLL, energy, Gamma CRPS, NULL Brier, 50/80/90/95/99% joint coverage, Gamma ranking, selected NULL and actual net value. Compare paired chemistry-group and layout bootstrap intervals conditional on saved predictions, without claiming independent certification or correcting repeated development.

Expected result before execution: separate scales may improve error scores; no direction or magnitude of acquisition benefit is presumed. Fixed first two moments do not imply an invariant expected nonlinear Gamma, so 'selection cannot change' is not a mathematical premise. A shape-only scale-mixture test is deferred until this comparison is read; direct conformal interval calibration would not certify individual NULL probabilities.

Only new executable paths receive unit tests: derivative, projector, exact Gaussian fallback, projected-likelihood equivalence, and positive scale fitting. Run the full declared object set after tests. Do not retrain or alter STATE50.
