# Sampling stability, then the matched basis-function contrast

Specified before new scores or fitting. This follows the 30-epoch Gamma-supervision
comparison on 639 already-opened source_5 DEV compounds.

## 1. Frozen-prediction Monte Carlo check

Keep the original A_HR, J_GEOMETRY_CONTROL and K_GEOMETRY_GAMMA_CRPS checkpoint
means and full nine-coordinate covariances. Rescore original ADD_TWO Gamma with
three new offsets (310000, 410000, 510000), each added to the existing fold seed.
Use common normal draws across arms within a fold and seed. Score 2,000 and
10,000 draws; the former is the prefix of the latter. First reproduce fold0's
original 2,000-draw scoring with the original seed.

Report Gamma CRPS, NULL Brier, Gamma mean and rank, and original physical-well
budget value/FDP/FPR with 79 selected compounds and 158 additional wells.
Selection is recalculated for each evaluation seed to reveal Monte Carlo
instability; neither the frozen model nor its loss is changed. The historical
primary evaluation remains unchanged.

Proceed to the basis-function contrast if all three 10,000-draw K−J CRPS
differences favor K and their mean magnitude exceeds twice the standard error
across the three Monte Carlo repetitions. This is a numerical continuation
condition, not a test of scientific generalization or certification. It does
not add independent compounds, training seeds or batches.

## 2. Same Gamma supervision, generic versus structured chemical responses

If the continuation condition is met, add one fresh arm:

| Arm | Chemical basis | Training loss | Action this round |
|---|---|---|---|
| A_HR | Original frozen HR | Historical | Copy unchanged |
| F_CONDITIONAL_GENERIC | Three RBF responses | Geometry | Copy old F unchanged |
| J_GEOMETRY_CONTROL | T, T², T⁴ | Geometry | Copy old J unchanged |
| K_GEOMETRY_GAMMA_CRPS | T, T², T⁴ | Geometry + Gamma CRPS | Copy old K unchanged |
| M_CONDITIONAL_GENERIC_GAMMA | Three RBF responses | Geometry + Gamma CRPS | Train five folds |

T is chemical Tanimoto similarity to the TRAIN anchors. The generic basis uses
RBF centers 0,0.5,1 and width0.25. Each chemical block keeps its previously fitted
TRAIN RMS scale. Both branches retain the same signed morphology and observed
scalar responses, local coefficients, 131→16→9 conditional MLP, bias-free linear
readout, frozen HR backbone and 3,837 new trainable parameters. The difference
therefore concerns the chemical-response basis (with its associated TRAIN scale),
not removal of chemistry or every type of biological prior.

M uses the existing `train_supervised` implementation and exact same configuration
as K: fresh matched initialization, 30 epochs/210 updates, batch64, AdamW0.0003,
weight decay0.0001, original 100-epoch learning-rate schedule with warmup,
clip5, geometry MSE plus0.1 increment penalty plus original Gamma CRPS divided
by TRAIN Gamma SD, weight1,64 independent training pairs. The same initialization,
permutation and training Monte Carlo seeds are retained. No coefficient, bandwidth,
loss weight or epoch is selected from this round's results.

M initialization must match old F's full parameter state and K's initial
trainable parameters. The four historical arms and all covariances remain intact.
Monitor epoch0/5/.../30 using the existing fixed128-pair readout, and evaluate the
actual epoch30 using the original 2,000-draw seeds for comparability with F/J/K.

## Comparison and interpretation

Main contrast K−M isolates the chemical basis under the same Gamma supervision.
M−F and K−J show the supervision effect for each basis; (K−J)−(M−F) is the
descriptive interaction for per-object CRPS, geometry MSE and Brier. M−A retains
the frozen baseline reference. Use the existing 2,000 paired within-fold compound
bootstrap comparisons and original budgets. Include every declared arm and fold.

If both bases benefit similarly from Gamma supervision, attribute the common
improvement to objective alignment rather than a special biological mechanism.
If K−M is uncertain, retain uncertainty instead of choosing a winner from point
estimates. A better probability score alone is not improved selection value or
purity. These are historical DEV comparisons, not independent certifications.

Original Gamma, the seven-part contract, original splits, FINAL, fifth repeats,
unopened candidates and old paused jobs remain unchanged. New outputs have fresh
run directories. No further epochs or additional search start automatically.
