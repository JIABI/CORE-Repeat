# Training-objective implementation checks

11 September 2026. These checks concern mathematical implementation and
execution, not the predictive performance of the three fitted arms.

## Test results

The full repository regression suite passed: **251 passed, 1 skipped**. The
skipped test requires an explicitly supplied legacy-data path; this experiment
uses the existing R2 full-control adapter and did not enter that legacy path.

New checks cover exact-NLL training, unchanged default ELBO, fair CRPS and its
gradients, original utility roles and costs, fixed outcome coordinates, shared
JEPA binding, identical initial states, separate auxiliary randomness, fitting
all arms before evaluation, and independently recomputed comparison metrics.

## Complete real-data objective path

The new C objective was exercised once on an actual 32-compound training
minibatch, using the 383-compound training library, all 3,617 coordinates, and
the complete 13,913,917-parameter architecture. NLL backward, 16-draw joint
utility CRPS backward, gradient clipping, and one optimizer update completed;
all gradients were finite. All 32 endpoints were scored. This untrained
execution check is not included in the three-arm efficacy comparison and its
weights are not reused.

On four CPU threads, this check took 14.02 seconds for the base NLL forward and
backward, and 3.93 seconds for the utility auxiliary forward and backward.
Peak process memory was approximately 5.32 GiB. These are startup measurements,
not a promise of full-run throughput.

## Equivalent masked-latent computation

The random-role likelihood previously retained latent columns belonging only
to padded, wholly unobserved wells. In one actual 32-compound batch, this formed
a 5,152-dimensional environmental factorization although only 448 columns
touched observations. Eliminating the remaining 4,704 identity-only columns
preserves the marginal likelihood and gradients.

The replacement was checked against the existing uncompressed implementation
and a dense numerical reference, including observed unknown groups and
currently zero-valued observed loadings. Future joint samples were unchanged.
No physical observations, feature coordinates, or model ranks were removed.
On the same full batch with two CPU threads, total step time changed from
50.39 to 23.01 seconds. The experiment retains four threads in all three arms;
this timing comparison was only used to diagnose the computational bottleneck.

## Data boundaries

Only the existing primary 639-compound DEV adapter was used. The source split,
four roles, original endpoint, original contract, old FINAL, and fifth repeat
are unchanged. No result for the fitted A/B/C comparison is available from
these implementation checks.
