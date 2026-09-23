# Fixed-structure continuation from epoch 10 to epoch 30

This continuation was requested after inspecting the opened DEV epoch-10 results. It is a training-budget development comparison, not a new independent validation or a pre-epoch-10 prediction.

## What remains fixed

Use the same 639 already-opened source_5 compounds, X/Z1/Z2/V, five folds, original target transforms, training-only response bases and scales. Retain A_HR unchanged. Continue all 15 B_MLP/C_GENERIC/D_STRUCTURED fits, not only promising folds or arms. The single-path architectures, loss, incremental penalty, full RIDGE_VALID joint error covariance, original Gamma and seven-item contract do not change. FINAL, fifth repeats, unopened candidates and old results remain untouched.

## Exact continuation

Restore each actual epoch10 checkpoint, including model and buffer states, AdamW moments and step counters, scheduler state, global Torch RNG and data-order generator RNG. Recreate the same scheduler function with the original 100-epoch horizon (700 steps per fold), not a new 30-epoch horizon. Do not restart warmup or initialize a new optimizer trajectory. Continue epochs11–30, taking140 additional updates for210 total. Copy the historical validation records and inherited validation-best checkpoint for the descriptive appendix; do not substitute it for the requested epoch30 scores.

All numerical model and scoring implementation files must match the prior source snapshot. A small uninterrupted-versus-resumed synthetic equivalence test checks continuation mechanics; it is not biological evidence. New experiment outputs use a fresh directory.

## Reporting

Save checkpoints15/20/25/30 and validation diagnostics every five epochs, retaining10. Score all arms at the actual epoch30 using the same2,000 joint-draw seeds, original three actions and 25% physical-well budget. The main ADD_TWO policy selects79 compounds and158 wells, within folds. Reuse the frozen HR predictions exactly.

Report six arm-to-arm comparisons at30 and within-arm paired30−10 changes using the same2,000 fold-stratified compound bootstrap draws. Preserve value and risk together: geometry MSE, Gamma CRPS, NULL Brier, association, selected actual net value, FDP and FPR. Show fit/validation trajectories and descriptive best epochs, but do not select a test-favorable epoch or change a threshold.

Interpretation distinguishes: continued validation improvement; training-only improvement; geometry improvement without utility improvement. Reaching30 is a fixed endpoint, not proof of convergence. No further continuation, new architecture, covariance re-estimation, JEPA, or mechanism annotations are enabled automatically.
