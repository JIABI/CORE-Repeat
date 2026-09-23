# Four-arm continuation from epoch 30 to epoch 60

## Question and scope

The user requested 30 additional epochs after examining the epoch-30 development results. Test whether further optimization changes the generic-versus-structured kernel contrast, with and without the existing Gamma loss. This is a continuation on the same opened DEV cohort, not a new independent validation.

Source: `runs/generic_gamma_epoch30_20260914_v1`. Use all 639 previously opened source_5 compounds and their existing four physical roles. Keep all five compound-grouped outer folds, inner partitions, original gains, seven-item contract, and physical budgets unchanged. Do not access FINAL, fifth repeats or unopened compounds.

## Arms

| Arm | Chemical response basis | Training objective |
| --- | --- | --- |
| F_CONDITIONAL_GENERIC | Generic RBF | Existing geometry loss |
| J_GEOMETRY_CONTROL | Structured | Existing geometry loss |
| K_GEOMETRY_GAMMA_CRPS | Structured | Existing geometry loss + normalized Gamma CRPS |
| M_CONDITIONAL_GENERIC_GAMMA | Generic RBF | Existing geometry loss + normalized Gamma CRPS |

A_HR is the copied, unchanged reference. The shared small conditioning MLP, morphology and amplitude bases, and bias-free readout remain unchanged. This comparison concerns the implemented chemical response bases, not the value of biological knowledge in general.

## Continuation

Resume each actual `epoch30.pt`, not its best checkpoint. Restore model, AdamW moments and step count, LambdaLR state, minibatch order RNG, Torch RNG and, for K/M, the dedicated training Monte Carlo RNG and objective buffers. Preserve the original 100-epoch scheduler horizon, warmup history, batch size 64, weight decay, loss weights and gradient clipping. Do not restart or extend the schedule. Each fit adds 210 updates, from 210 to 420. Stop all arms at epoch 60, without automatic extension.

The HR backbone, response bank, landmark identities, descriptor scaling, target coordinate transforms and full joint residual covariance remain frozen. Only the same 3,837 branch parameters continue learning. Save every five epochs. Retain the original histories and checkpoints without changing the source directory.

## Readout

The primary comparison is fixed actual epoch 60 against actual epoch 30. No outer-fold result selects a checkpoint. Report inner-validation trajectories at epochs 30/35/40/45/50/55/60 as descriptive optimization diagnostics; do not label a single minimum convergence.

Use the same 2,000 joint predictive samples and original per-fold random seeds for all epoch-60 evaluations. Keep the existing 25% physical-well budget, selecting 79 compounds in total (158 extra wells). Report geometry MSE, Gamma CRPS, Brier, rank correlation, realized selected mean gain, NULL counts, FDP and false-activation rate. Compare K versus M, J versus F, K versus J and M versus F, and the factorial score interaction. Also report all four within-arm 60-minus-30 changes and the unchanged A reference.

Use the existing 2,000 paired within-fold compound bootstrap replicates. These conditional, unadjusted intervals do not incorporate training, batch dependence, Monte Carlo uncertainty or repeated development selection. The earlier 10,000-draw check established K-versus-J sampling stability at epoch 30 only; it does not certify new epoch-60 contrasts.

## Verification

Before execution, test exact restart equivalence for both objectives, including optimizer and RNG states. During execution, verify frozen tensors, identities, objective buffers, checkpoint step counts and scored checkpoint reproduction. Stop on numerical or identity failures; do not substitute another model or omit a failed arm. Keep previous outputs intact.
