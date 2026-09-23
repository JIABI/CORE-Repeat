# JEPA optimization diagnostic: batch 128 and decaying learning rates

## Purpose and scope

This is a development diagnostic of representation training, not a formal
OPAL efficacy experiment. It asks whether a lower learning-rate trajectory
improves conditional prediction after accounting for teacher drift and
representation scale. No world-model training, acquisition evaluation,
contract certification, or automatic promotion of a checkpoint follows it.

The earlier objective-comparison process (PID 89675) was suspended on
2026-09-11, after its A/ELBO epoch-14 checkpoint. Its existing files and source
snapshot are retained. Its old status.json predates this suspension; the
process state and this record describe the pause. Do not automatically resume
it when this diagnostic finishes.

## Data and full method

- Existing source_5 primary DEV archive only: 639 compounds, four roles,
  3,617 fixed measurement coordinates.
- Preserve the existing split: 383 training and 96 validation compounds.
  The 64 calibration and 96 evaluation compounds are not scored, fitted, or
  used for model selection in this diagnostic. The old OPAL FINAL and fifth
  repeats are not accessed.
- The existing 639-compound archive is loaded as one file, including the
  existing DEV calibration/evaluation arrays. Their physical presence in
  memory is not described as an unopened partition; they are not used by the
  diagnostic's fitting, scoring, or selection procedures.
- Fit all scaling, reference templates, and the library bank on the same
  383 training compounds. Use the original information-availability masks.
- Keep the complete conditional JEPA: CellProfiler feature-group attention,
  hidden dimension 256, two attention layers, four heads, chemical context,
  legal reference context, library context, student/projector/predictor and
  EMA teacher.
- Preserve the current loss: 25 alignment + 25 variance-floor penalty +
  covariance penalty. The first comparison diagnoses optimization without
  simultaneously replacing the representation objective.

## Paired learning-rate comparison

| Setting | R1_COSINE_3E4 | R2_COSINE_1E4 |
| --- | --- | --- |
| AdamW starting learning rate | 0.0003 | 0.0001 |
| Final learning rate | 0.000003 | 0.000001 |
| Schedule | Monotone cosine decay | Monotone cosine decay |
| Epochs | 60 | 60 |
| Compound batch size | 128 | 128 |
| Weight decay / gradient clipping | 0.0001 / 5.0 | 0.0001 / 5.0 |
| Initial tensors / episode stream | Identical | Identical |

Learning rate is scheduled per optimizer update, from the declared maximum
to 1% of that maximum. There is no warmup in this comparison: both trajectories
decrease from the start, and they differ only by a factor of three. Run arms
interleaved by epoch so both produce progress in each reporting interval.
The final batch has 127 compounds; no compound is dropped or duplicated.

The historical batch-32/constant-rate run is descriptive context, not an
isolated test of batch size or scheduling. Both change relative to that run.
This paired comparison isolates the learning-rate amplitude at batch 128.
Sixty epochs at batch 128 have fewer optimizer updates than sixty epochs at
batch 32; comparisons to the old run must acknowledge that difference.

## EMA clock

EMA controls the teacher's smoothing time, not a separate supervised label.
Keep its reference decay at 0.99 per 32 compound exposures. For a batch with
n compounds, use beta = 0.99 ** (n / 32). Thus a 128-compound update uses
0.960596, and a 127-compound update uses its corresponding exponent. This
preserves the reference exponential forgetting rate per compound exposure;
it does not make different batch-size optimization paths identical or claim
that this EMA time constant is optimal. Use the same rule in both arms.

## Monitoring and checkpoints

- Lightweight epoch aggregates: objective and all three loss components,
  actual learning rate, EMA decay, unclipped gradient norm, clipping rate,
  completed batches and elapsed time. These make progress inspectable.
- Detailed diagnostics at initialization, epoch 1, every 5 epochs, and the
  final epoch. This is the reporting/checkpoint cadence, not an early-stop rule.
- Fixed evaluation tasks: X role 0 predicts roles 1, 2, 3, with the same
  validation compounds and masks at every diagnostic. Also evaluate a fixed
  training subset of 96 compounds chosen by identity, not by outcomes.
- Report teacher-target raw MSE, centered target variance, their ratio (NMSE),
  cosine similarity, and teacher/student same-input drift. A zero denominator
  produces an explicit undefined score, not a fabricated perfect prediction.
- For student projected embeddings, student encoder outputs, teacher projected
  embeddings, teacher encoder outputs, and predictions, report coordinate
  standard deviations, near-zero dimensions, covariance participation rank,
  entropy effective rank, and sample counts. State the finite-sample rank cap.
  Teacher encoder outputs are the representations actually exported downstream.
- On a fixed training task, inspect each weighted loss term's gradient norm
  and their directional agreement on the shared student encoder. Do not update
  weights, teachers, running statistics, or training random streams during these
  diagnostics.
- Save the full learner and optimization state at diagnostic checkpoints.
  Keep initialization and periodic checkpoints. Do not select epoch 5 from the
  historical run merely because its unnormalized total loss was smallest.
- Report every ten minutes in the current task. Reports describe actual
  completed epochs and changes since the preceding report. If no new detailed
  diagnostic is available, distinguish lightweight progress from detailed
  evidence. Report exceptions without silently changing settings or restarting.

## Memory implementation

Batch 128 remains a joint batch for the loss, including its global covariance.
Encoder activation checkpointing and profile-wise computational chunking may
reduce retained activations. They do not reduce feature dimension, attention
depth, representation dimension, reference inputs, library inputs, or the
statistical batch. Validate full-loss and gradient agreement with the original
encoder before the real-data diagnostic. Ordinary accumulation of four
independent batch-32 covariance losses is not equivalent and is not used.

## Reading the outcome

A falling raw loss alone is not success. Interpret it alongside normalized
prediction, teacher scale, effective rank, and train–validation differences.
Both predictions and their learned teacher targets may change; NMSE and cosine
are useful diagnostics, not certificates of biological utility. A lower-rate
arm that improves validation prediction while retaining representation spread
would support an optimization contribution. A similar normalized trajectory
with different raw scales would support a scale/drift explanation. Neither
outcome alone establishes improved acquisition decisions.

If regularization gradients dominate or conflict with conditional prediction,
the next separate diagnostic can test train-fitted target standardization or
regularization-weight changes. Do not quietly introduce either into this pair.
Splitting shared predictable structure from irreducible measurement variation
is another modeling question; the present run does not resolve it by forcing
cross-repeat invariance.
