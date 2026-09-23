# R4 execution: independent-identity follow-up evaluation

21 September 2026. The author requested all R4 experiments. The execution uses the existing task, models and evaluation draft; it does not add biological or representation models. This file fixes the numerical recipe before final fitting. Confirmation access additionally requires the identity qualification record and the staged access files.

## Primary comparison

Compare CORE_ORIGINAL with access-matched HistGB regression plus its previously specified calibrated NULL classifier at the same budget. CORE is the unchanged RIDGE, validation-selected HR, A30 and STATE50 mean with AMP_EMP_LOCAL; its base scatter remains the RIDGE grouped-out-of-fold second moment. Do not replace it with the recent cross-dose response models or choose another covariance after confirmation.

Both scores use expected Gamma minus 0.2 times P(NULL). Gamma is one half of the cosine improvement from averaging X, Z1 and Z2 against V, minus 0.02 for two action wells. The primary contrast is CORE minus HistGB total realized Gamma divided by the full metadata-qualified N. The scientific planning margin is an additional 0.01 mean Gamma per selected object; it is not a non-inferiority margin or a guaranteed detectable difference.

Prespecified secondary policies are lambda=0 for the same predictors, the matched-mean Gaussian joint model, uniform same-budget random selection and its exact finite-cohort expectation, a constant-score stable-ID policy, STOP and all-eligible acquisition. The last two are budget backgrounds, not same-budget competitors. The direct classifier probability is a separate risk readout; do not claim it and the direct regression-residual distribution are jointly coherent.

## Final fit and scoring

The 904 already-open development identities are allocated once by sorted-ID PCG64 permutation with seed 20260921: TRAIN434, VALIDATION108, REF_FIT181 and DIST_CAL181. IDs and groups are listed in development_partitions.json. This is final fitting, not a new claim of independent development performance. The feature space reuses the existing DEV904 control-derived transform and coordinate mask.

The complete CORE training budgets and existing HistGB hyperparameter/CAL procedures are retained. No confirmation identity contributes to model fitting, reference construction, probability calibration or external-station anchors. Once the final models are saved, no fitting resumes after confirmation X is opened.

For qualified N, B=floor(N/4) action wells and k=min(floor(B/2), number of valid X objects). Missing X does not reduce N. All methods use the same eligible set; ties use ascending stable identity. The author authorized the final 1539-identity cohort after exclusion of EOS101686 (MS023) and EOS101275 (CGI-1746), in addition to the eight previously established overlaps. B=384 and k=192 if enough valid X objects exist. The original 1549-identity ledger and all ten exclusion reasons are retained in qualification/original1549_disposition.csv.

Each joint model uses 100,000 draws per object. Seeds are 20260921, 20360921 and 20460921; only the first determines the main selected list. The others quantify numerical changes in lists and NULL counts and never select the reported winner. Main-seed Gamma draws are cached in float64; full nine-dimensional draw arrays are not cached because of available storage. Any later deterministic replay for secondary observable scores reuses the same frozen distributions and seeds, not a new fitted model or selected policy.

## One evaluation table, multiple readouts

After the lists are fixed, compute Gamma/NULL once for all qualified candidates with observable outcomes. Reuse the ID-aligned table for value, selected mean Gamma, NULL count, FDP, FPR, sensitivity, expected versus observed selected NULL, global Brier/AUC, selection-band reliability, distribution scores and intervals, overlap, random-policy reference, Monte Carlo sensitivity and costs. Missing endpoints remain missing with the existing bounded-outcome arithmetic; they are not silently dropped from N or imputed as observed NULLs.

The realized finite-campaign contrast is primary. Chemical-group and seven-library-layout resampling, 10,000 replicates with seed 20260923, describe dependence sensitivity. For a reconstructed campaign, repeat frozen top-k with its reconstructed N; fixed-list contribution resampling is reported separately. Leave-one-layout results are descriptive. These are not iid-binomial certificates or guarantees for arbitrary future sites.

Main costs assume an existing fitting/reference/calibration pool. Also report from-scratch REF, REF+CAL and total fitting-resource acquisition, with amortization over 1, 2, 5, 10 and 20 campaigns. Count each physical well once. Gamma already includes action cost; evaluation wells and computation are separate accounts.

## Secondary screening-use endpoint

Use the pre-existing MEDINA/USC plan: determine whether averaging the two additional FMP wells improves agreement of morphology-neighbourhood rankings with those two sites. The anchor identities come only from DEV904, with station-specific DMSO preprocessing fixed before confirmation X. Both external sites are reported. If metadata access or a usable development-defined anchor space cannot be established, record that endpoint as unavailable before seeing confirmation outcomes; do not choose a different endpoint because of its measured performance.

## Execution order

1. Record final development roles, implement and test the distinct fit/score/evaluate paths. Finish historical identity follow-up and exact candidate qualification. Final development fitting can run while this metadata work is completed.
2. Save the qualification and final protocol in freeze.json. For unresolved identity uncertainty that prevents the intended independent-identity claim, request an explicit scope decision before releasing confirmation data.
3. Export confirmation X with the frozen transform, retaining all qualified IDs and missingness. Fit no feature choices from their outcomes. Score once and save every prespecified list in SELECTIONS_FROZEN.json.
4. Only then export FMP future-role measurements and the predefined external-station validation profiles. Produce the complete endpoint table and all reports; do not tune, replace objects or reselect a strategy.

The user request authorizes this staged workflow. It does not bypass the statistical separation between fitting, decision input and endpoint evaluation, and it does not release unrelated protected cohorts.
