# Selection-matched stability of bounded conditional-geometry correction

## Question

Does the previous bounded neural mean correction improve original ADD_TWO
prediction and same-budget selection after matching access to validation
labels, and does that improvement persist across partitions and training seeds?
This is further development on the 639 already opened DEV compounds. It is
not a new independent test, an original-contract certificate, or a complete
cross-site biological hierarchical model.

## Fixed data and repetitions

Use the existing four roles X/Z1/Z2/V and all 3,617 X features plus X log norm.
Keep the exact original measurement-space utility, all three actions, costs,
technical completion rules and seven-item contract. Do not read FINAL, the
fifth repeat, or unopened compounds. No outcome clipping, exclusion or new
outcome-based stratification is introduced. Original files remain unchanged.

Partition seeds are 20260914, 20260915 and 20260916. Each assigns five outer
test folds by the existing X-log-norm-stratified algorithm; only X and IDs
determine assignment. Each outer training pool is split into fit and external
inner-validation using the same 20% validation algorithm. The first assignment
must reproduce the preceding run exactly. Every compound has one test
prediction per partition, not nine independent observations.

Each fold fits three residual networks at seeds fold_seed+401, +1401 and +2401.
All are reported; neither seeds nor partitions are selected using results.
The old HR result remains a historical reference. Because the ridge underneath
the new HR now uses matched external selection, this is not a bitwise repeat
of the old HR recipe even in the first partition.

## Six arms and selection information

1. GLOBAL_GEOMETRY: one joint Gaussian, fit on the fold's fitting outcomes.
2. RIDGE_TRAINCV: the preceding complete fit-only five-outer/four-inner ridge
   selection and OOF error-covariance procedure, refitted for each partition.
3. RIDGE_VALID: the same ridge path, but lambda in {0.01,0.1,1,10,100} is
   selected using the external inner-validation objects. The criterion is MSE
   in the fit-only standardized nine-coordinate frame; numerical ties choose
   larger lambda. Validation objects do not fit coefficients or preprocessing.
4–6. HR_VALID_S0/S1/S2: fixed RIDGE_VALID plus the unchanged bounded residual
   network, one per training seed. All three retain exactly RIDGE_VALID's joint
   covariance, isolating the mean-learning procedure.

For RIDGE_VALID covariance, five internal error folds partition fitting
objects. Each clone fits preprocessing and its ridge path on the other four
folds, chooses lambda on the same external inner-validation set, then predicts
its excluded error fold. Restore predictions to native coordinates before
computing errors in the final fit-only standardized frame. Use the full
Ledoit–Wolf centered error covariance plus the residual-bias outer product.
It replaces rather than supplements any covariance. It represents predictive
error, not an identified biological or technical variance component.

RIDGE_VALID and HR have the same fit/selection/test objects, input coordinates,
and validation metric. Their search families and number of validation checks
are not identical. HR uses validation sequentially for the backbone penalty and
its checkpoint; matching label availability does not prove equal selection
complexity or make the comparison independent of historical development.

Do not rerun G/L, add a new HR covariance branch, activate JEPA/biological
kernel, or change the target in this test. Their old results are not overwritten.

## Unchanged complete residual fitting

Fixed ridge plus D+1 -> 32 -> 32 -> 9 GELU network, dropout 0.1,
zero-initialized final layer, correction 0.5*tanh(output), float64.
Loss = mean MSE + 0.1 times squared correction. AdamW learning rate 0.0003,
weight decay 0.0001, batch 64, gradient clipping 5, 30-step warmup and cosine
decay to 0.000003. Maximum 200 epochs; check external validation MSE every
five epochs including zero; retain the best checkpoint. Stop after eight
checks without improvement of 0.00001, no earlier than epoch 40. Record
every epoch and save a report every 20 epochs. No extension after early stop.

## Evaluation and interpretation fixed before execution

Use 2,000 legal joint Gaussian-coordinate draws per object and the existing
factor-forward Gram decoder. Within each fold all conditional arms share the
same integration noise seed; GLOBAL uses identical broadcast draws to avoid
ranking identical laws by Monte Carlo noise. This is an integration coupling,
not a claim of physical dependence across compounds.

Primary comparison: average per-object ADD_TWO Gamma-CRPS of HR versus
RIDGE_VALID. First compute each seed/partition's score, then average scores
within compound. Do not mix distributions and score the mixture as if an
ensemble had been deployed. Also report NULL Brier, bias, rank association,
the same-coordinate mean MSE, geometry and coverage diagnostics.

Key secondary policy: original ADD_TWO expected-gain ranking at the 25%
physical-well cap, selected within folds, totaling 79 compounds/158 wells in
each partition/seed. Report realized net value, NULL count, FDP, FPR and
sensitivity jointly. Keep all original 5/10/25% budgets and action reports;
do not replace the principal readout with a more favorable budget.

Report all nine partition/seed comparisons and all 45 fold comparisons, plus
the three partition-level seed averages. Include RIDGE_VALID minus
RIDGE_TRAINCV to measure the change associated with validation-based selection.
Assess selection overlap and concentration without deleting difficult objects.

Conditional paired intervals resample the 639 IDs jointly after averaging
their score/contribution differences over repetitions and seeds, using 2,000
bootstrap resamples. They do not treat 5,751 repeated predictions as independent
or convert repeated activations into an increased certification sample size.
They hold fitted models and selection masks fixed and do not cover training
overlap, shared experimental batches, historical search or Monte Carlo error.
Show empirical seed/partition variation separately rather than presenting
these conditional intervals as a formal generalization guarantee.

Interpretation: stable mean improvement alone supports mean learning, not
selection. Stable policy value improvement with its accompanying risk supports
further strategy development, not original-contract authorization. An overall
distribution-score claim additionally requires the primary comparison to
support it. Mixed directions or intervals spanning zero remain inconclusive;
they do not trigger new hyperparameter searches in this run.

## Execution

Prepare all memberships and this protocol before training, execute a recorded
source snapshot in a fresh run directory, and preserve every completed result.
Run sequentially with two CPU threads. The run has 15 outer folds and 45
residual fits; it ends after the fixed queue or an actionable numerical failure.
No failed run is silently restarted and no old paused training is resumed.
