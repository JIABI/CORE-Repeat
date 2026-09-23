# Step 1: fixed-model replicate-structure diagnosis

Question: Is the observed undercoverage confined to held-out compounds, or
already present on the fitting compounds? Which aspects of the future joint
distribution fail: individual wells, pair differences, or well averages?

Use only the existing source5_primary_fullcontrols export: 639 already opened
DEV compounds, four physical roles, 3617 unchanged coordinates. Preserve the
383/96/64/96 split and slot-0 X / slots-1,2,3 future roles. No FINAL, fifth
repeat, new compounds, raw images, custody, or external cache is accessed.

Load seed20260912 A and the already fitted full-space L from the completed
moment_repair_20260913_v1 run. Do not refit a scaler, covariance, model, or
threshold. Main comparisons are A and L. Retain the previous mean-only M and
mean-plus-within Q as frozen component diagnostics; no new ablation is trained.

For each model and all four partitions, compute exact Gaussian linear
contrasts for the three single future wells, three pair differences, three
pair means, and the mean of all three future wells. Use the full block-aligned
covariance, including retained shared environments, and the unchanged physical
coordinate transform. This is exact evaluation of the existing Gaussian, not
a reduced model or Monte Carlo approximation.

Report coordinate coverage, width and interval score at 50%, 80%, 90%, 95%;
residual bias, RMS, standardized residual second moment, and realized residual
RMS / model standard deviation (explicitly not coverage). Keep per-compound
traces and all objects. Summarize typical and extreme-object behavior together.
The analysis does not identify pure biological/technical variance components.

Validate identity and mean predictions against saved held-out outputs, and
compare analytic coordinate/difference coverage to prior finite-Monte-Carlo
intervals. The former should match up to floating-point effects; the latter
need not be exactly equal because the old intervals used 2000 draws. Where
available, also compare the stored per-compound density evaluation.

Preserve the old complete Gamma, NULL and policy outputs as the nonlinear
decision checks; do not rerun their Monte Carlo or reinterpret training scores
as deployable performance. No policy search or certification is performed.

Train-versus-held-out differences are diagnostic evidence, not a binary proof
of overfitting versus nonidentifiability. Train inference uses the same legal
context and fitted reference/library access; self-compound library entries
remain excluded. Historical validation was used for checkpoint selection.
Shared batches and 3617 correlated coordinates do not constitute independent
replication. No binomial confidence claim is made from coordinate counts.

Run on two CPU threads without stopping, resuming, or changing other jobs.
Stop after the diagnostic report. Heteroscedastic c_i, heavy tails, new neural
training and grouped cross-validation are subsequent decisions, not this run.
