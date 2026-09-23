# HR with a chemistry–phenotype kernel: epoch-10 development comparison

This experiment asks whether a structured response branch adds useful information to the existing HR geometry mean. It is an early development comparison, not a completed training comparison or a new certification experiment.

## Data and frozen baseline

Use all 639 already opened source_5 DEV compounds and only X, Z1, Z2, V. Reuse the first five-fold partition and HR_VALID_S0 checkpoints from hierarchical_stability_20260914_v2. The five test folds contain 128, 128, 128, 128, and 127 compounds; each compound is evaluated once. Preserve the original endpoint, costs, seven-item contract, original split files, FINAL, fifth repeat, and unopened compounds.

The ridge coefficients, HR network, fit-only preprocessing, and full joint nine-coordinate ridge error covariance remain frozen. Restore the saved HR checkpoints directly; do not retrain them. New branches see the same complete decision-time X and chemical inputs. They never use future query wells to construct features, anchors, or support statistics.

## Three arms

- A_HR: the exact saved HR prediction and joint covariance.
- B_GENERIC: HR plus a generic fixed-RBF response branch.
- C_STRUCTURED: HR plus the Tanimoto-locality basis [1,T,T^2,T^4].

Both new arms use the same training-only chemical landmarks, chemistry–phenotype descriptors, KAN mixing architecture, output size, optimizer, data order, and selection budget. Generic RBF centers come from training landmark descriptors; three bandwidths are fixed at 0.5, 1, and 2 times a positive training-distance scale. Parameter counts are recorded and compared. This is a chemical–phenotype locality hypothesis, not a validated target/pathway mechanism or pharmacodynamic law.

Descriptors retain chemical similarity, normalized phenotype-direction similarity to the same landmarks, the existing X log-norm coordinate, transformed-X magnitude, and fingerprint density. At most 64 chemical anchors are fitted using only the current fold's fit IDs. Descriptor scaling and RBF construction use fit objects only. Unknown chemistry is masked and falls back to HR. Neighbor support is reported with self-exclusion for training identities, but is not converted into an outcome-tuned threshold: the first comparison uses an availability gate and a penalty on the actual mean increment.

## Mean-only connection

mu_new = mu_ridge + b*tanh(f_HR(X) + available*f_kernel(descriptors)), with the existing b=0.5.

The kernel's nine-output head starts at zero, reproducing HR exactly. Freeze HR parameters and keep its dropout in evaluation mode throughout kernel training. Do not reuse the old latent-prior mean/variance heads. The total correction bound stays unchanged. Penalize the actual new mean minus HR, not the old frozen correction. Keep the identical ridge covariance for all three arms, then decode through the existing legal Gram construction and original Gamma calculation.

Monitor original-HR tanh saturation, kernel gradient norms, mean increment size, support, and frozen-parameter identity. A branch that cannot move because of saturation or unavailable inputs is not evidence that chemistry contains no information.

## First-stage training and reporting

Use one fixed new training seed per fold, equal across B and C; seed = the fold's recorded seed + 7401. AdamW learning rate 0.0003, weight decay 0.0001, batch size 64, gradient clip 5, incremental penalty 0.1, KAN hidden width 24, mixing width 32. Warm up for 10 optimizer steps and follow a cosine schedule defined over 100 epochs, minimum learning rate 0.000003. The first authorized stage stops at epoch 10, rather than compressing an entire convergence schedule into 10 epochs.

Check fit and inner-validation geometry MSE at epochs 0, 5, and 10. Save the exact epoch-10 model and complete optimizer/scheduler/RNG state. Retain the validation-best checkpoint among 0/5/10 as supplementary information. The primary early comparison uses epoch 10 for both new arms regardless of which arm or epoch looks better. Test results do not select an epoch or change the other arm. Report negative, null, and positive results alike; do not wait for a favorable fluctuation.

Use 2,000 joint draws and the same fold-specific random stream as the frozen baseline. Keep the existing outcome-preserving geometry evaluator, including numerical factor verification. No sample replacement, outcome clipping, or object exclusion is introduced.

## Readouts

Report nine-coordinate mean MSE, original ADD_TWO Gamma CRPS, NULL Brier score, descriptive Gamma rank correlation and AUC, geometry diagnostics, and realized value/risk under the original fixed budgets. The main 25% budget is a physical-well budget: each test fold selects floor(floor(0.25*n_test)/2) compounds, giving 79 compounds and 158 additional wells overall. Do not pool the folds and rerank globally.

Compare B-A, C-A, and C-B using paired per-compound scores and fixed selection contributions, with 2,000 bootstrap draws. These are conditional development intervals, not independent certification bounds; the five folds share batches and overlapping training sets. Additional information and capacity both change from A to B/C. C-B isolates the basis choice more closely. An improvement in geometry alone does not establish improved Gamma probabilities, biological mechanisms, or certified acquisition.

After the epoch-10 report, stop and discuss whether further training or a separate joint-error comparison is warranted. Do not start JEPA, change covariance fitting, activate sparse mechanism annotations, or resume older paused training as part of this stage.
