# M3 confirmation dispersion extension

22 September 2026, specified while the first development cells were fitting,
before fitting or evaluating any new confirmation control. The user requested
the predicted-dispersion comparator in confirmation as well as development.
The development protocol and all 60 running cells remain unchanged.

Fit the declared full-input HistGB log-W regression using the same final
DEV904 TRAIN434+REF_FIT181 and VALIDATION108 roles, the original final
TRAIN-fitted input transform, and the original chemical descriptors. Use the
same three configurations, iterations, learning rate, L2, selection criterion
and seed 20260921 as the amplitude protocol. DIST_CAL is not used by this
dispersion-only ranking. Score the 1,527 eligible confirmation X inputs,
rank by predicted log W descending, and select 192 with ascending-ID ties.
Freeze this new list in its own M3 checkpoint before looking up its outcomes.

Evaluate it with the same Gamma/NULL and paired external endpoint summaries,
signed missing-outcome bounds, overlap and chemical/layout resampling as the
other post-hoc controls. Its model has full first-well and chemical inputs;
it is not an amplitude-only arm. No confirmation outcome chooses its direction,
features, regression target, grid or hyperparameters. Original R4 files remain
read-only. The new dispersion checkpoint extends the existing M3 controls;
previously fitted controls are reused without refitting.
