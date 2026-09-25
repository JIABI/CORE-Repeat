# Reproduction guide

## The portable computational interface

The current model separates mean fitting, distribution fitting, and acquisition evaluation. The feature preparation and identity allocation are explicit inputs, not inferred by the fitting functions.

`opal2.eu_core_training.fit_complete_eu_core(model_train, model_validation, reference_fit, metadata, output, seed=...)` expects:

- `MODEL_TRAIN` and `MODEL_VALIDATION`: mappings containing exactly `ids`, `groups`, `Y`, `chem`, `chem_mask`.
- `REF_FIT`: the same mapping with `X` instead of `Y`; no future reference wells enter this mean-fitting interface.
- `Y`: `[N, 4, D]` finite profiles in `X, Z1, Z2, V` order, in the declared assay feature space.
- `X`: `[N, D]` finite first-well profiles in that same space.
- `chem`: `[N, 513]`, comprising 512 Morgan bits and the structure-validity indicator; `chem_mask`: `[N]` Boolean.
- `groups`: complete chemical-identity groups. No group crosses the supplied fitting/validation/reference roles.
- `metadata['chemical']`: actual fingerprint construction metadata.

The unchanged recipe needs 64 distinct reference fingerprints and available training chemistry. It is not designed to silently fabricate these inputs for a new archive. Assay feature filtering, plate control normalisation, missingness, identity mapping and role assignment are scientifically relevant processing choices.

`opal2.eu_core_distribution.fit_eu_distribution(...)` then uses the supplied reference and disjoint distribution-calibration residuals. It returns a conditional scatter/radial law without changing the fitted mean. `predict_eu_distribution(...)` accepts only first-well information plus the predicted nine-coordinate mean. Relevant docstrings and tests specify exact schemas.

`opal2.gram_geometry` converts genuine four-well profiles to the joint coordinates and evaluates the original one-/two-well half-cosine gains. These conversions are exact within the documented non-singular domain; they do not reconstruct cell images from geometry.

## Three reproducibility levels

| Level | Inputs | What is reproduced |
|---|---|---|
| Saved-result analysis | Companion `source_data/` and released research summaries | Paper figures, selected-list statistics, tables and specified sensitivity calculations |
| Frozen prediction | Final model, compatible prepared first-well arrays, fixed inference settings | CORE/direct predictions without using future outcomes or refitting |
| Full experiment refit | Prepared four-well data, metadata, frozen role manifests, reference inputs and predecessor artifacts specified by each runner | The full training and evaluation chain |

A saved-result plot is not a new fit. Passing unit tests is not evidence that all original experiments have been rerun on a new operating system. The release validation report names exactly which checks were executed.

## Companion archive layout

The companion archive's `research/` tree retains the original project-relative input paths. In particular:

```text
research/
  data/source5_primary_fullcontrols/
  data/lincs_pilot1_biology_20260915/
  data/rxrx3_r2_20260918/prepared_r2/
  reports/eu_core_development_20260917_v1/prepared_data_cc904/
  runs/...
  reports/...
```

For dataset-specific historical runners, create a **working copy** of the relevant `data/`, `reports/` and `runs/` children beside `opal2/` and `scripts/`. Do not point a write-producing runner at your only archived copy. The code uses repository-relative paths and has no dependency on the original author's home directory. External legacy loaders accept explicit paths (`legacy_root`); those older upstream project directories are not implicitly found or bundled.

For figure/statistical reproduction, follow `paper/` instructions instead. It reads the released source tables and does not require reconstructing every predecessor training directory.

## Final EU model

The explicit training wrapper accepts a **directory**, not an NPZ filename:

```bash
python scripts/fit_r4_model_20260921.py --help
python scripts/score_prepared_queries.py --help
```

The prepared development directory contains `data.npz` and `metadata.json`. The final fitting interface requires the recorded 904 development objects, split into 434 training, 108 validation, 181 reference and 181 calibration objects, and the frozen seed 20260921. To study another population, use the underlying array APIs with an independently declared evaluation design; do not relabel a new population as the original confirmation.

Frozen inference loads the following model files only:

```text
status.json
manifest.json
mean/preprocessing.json
mean/STATE50/epoch50.pt
distribution.joblib
histgb.joblib
```

The model files are serialised Python/PyTorch objects and should only be loaded from a trusted source. They include training-derived information and remain subject to any applicable upstream data conditions described in the companion deposit.

The `score_prepared_queries.py` wrapper forwards to the original `r4_final_model.score` implementation. It retains 100,000 draws/object, three seeds, chunk size eight, the declared risk weight and both joint-law diagnostic arms. It can save several gigabytes of Gamma draws for a full cohort. Its output must be a new directory. Raw profiles from a different site cannot be supplied merely because they have similar feature names: use the frozen feature order and original assay transform, or define and validate a separate adaptation protocol.

## Historical runners and operating systems

The dated runners document how the experiments were executed. For orientation:

| Analysis | Runner or module |
|---|---|
| First-well predictability | `scripts/run_r1_completion_20260917.py` |
| Four-resource model comparison | `scripts/run_r2_core_comparison_20260917.py`, `run_jump_r2_completion_20260918.py`, `run_lincs_r2_completion_20260918.py`, `run_rxrx3_r2_completion_20260918.py` |
| Biology/representation comparison | `scripts/run_r3_eu_modules_20260918.py`, `run_r3_rxrx3_modules_20260920.py` |
| Cross-dose response and relation diagnostics | `scripts/run_r3_crossdose_response_20260920.py`, `run_r3_jointdose_program_20260921.py`, `diagnose_r3_relation_variance_20260921.py` |
| Frozen confirmation | `scripts/run_r4_confirmation_20260921.py`, `opal2/r4_primary_analysis.py` |
| Conditional-quantile HistGB | `scripts/run_c3_quantile_histgb_20260922.py` |
| Amplitude-only controls | `scripts/m3_amplitude_controls_20260922.py` |
| Dependence ablation | `scripts/run_m4_dependence_ablation_20260922.py` |
| EU task-matched measurement comparison | `scripts/run_eu_direct_measurement_20260924.py`; `protocols/EU_direct_measurement_20260924.md` |

Many runners validate their specific original cohort sizes, fixed configurations, input schemas and saved-state history. Those checks are deliberate and were not disabled to make arbitrary inputs run. Some older experiments execute a copied `source_snapshot` and compare it with the source path in the run manifest. They need their original preparation step or a documented reconstruction of its input tree.

## EU direct-measurement comparison and geometry checks

Main Table 3 and Supplementary Note 16 compare frozen CORE with models trained directly on three nonnegative scalar observables, `log(1 + ||(a+b)/2||² / ||X||²)`, for `(a,b) = (Z1,Z2), (Z1,V), (Z2,V)`. CAL is the primary comparator here because both CAL and CORE use the disjoint distribution-calibration labels; RAW is secondary. This post-hoc comparison uses 1,520 complete confirmation objects and does not change any original CORE fit or acquisition list.

The companion `source_data/eu_direct_measurement_20260924/` contains the finished summaries, paired intervals and per-object predictions. Check those alongside the rest of the release without fitting:

```bash
export OPAL2_DATA_ROOT=/absolute/path/to/CORE-Repeat-data
python paper/verify_release_data.py
python -m pytest -q tests/test_gram_geometry.py \
  tests/test_measurement_forecast_replay.py \
  tests/test_observable_quantile_distribution.py
```

The geometry tests check agreement with explicit profile calculations, invariance to common rotation and scale, and different measurement observables under identical two-well gain. These are software checks of the representation; they are not additional empirical results.

For exact historical re-execution, use a separate writable analysis checkout. Copy the companion `research/reports/eu_core_development_20260917_v1/prepared_data_cc904/` and `research/runs/r4_confirmation_20260921_v1/` to the corresponding `reports/` and `runs/` paths there. Keep scikit-learn at **1.9.1**. The following is a full computation, not a quick release check:

```bash
python scripts/run_eu_direct_measurement_20260924.py --help
python scripts/run_eu_direct_measurement_20260924.py run
```

`run` fits 171 quantile regressors, replays 100,000 frozen CORE draws per eligible object, and evaluates 10,000 paired resamples. It writes `runs/eu_measurement_direct_20260924_v1/` and `reports/eu_measurement_direct_20260924_v1/`. The unchanged frozen prediction directory must contain its query metadata, coordinate preprocessing, query-specific distribution, prediction manifest and cached original Gamma draws. Saved direct-model checkpoints and predictions are also supplied under the companion `research/runs/eu_measurement_direct_20260924_v1/`; copying these into the working tree permits the runner's existing resume/evaluation modes. The `evaluate` action recomputes paired intervals from finished predictions without fitting or replay, and still requires the matching recorded environment and protocol.

The measurement runner uses POSIX file locking (`fcntl`) and the two-worker foreground process pool. Its complete historical run was not repeated during this source-folder refresh.

The numerical package is Python. Several orchestration helpers additionally use POSIX process tools, `fcntl`, `resource`, or macOS `caffeinate` for background execution. Use foreground execution where offered. The release does not claim that those historical background launch modes are portable to Windows or have been retested on Linux. Author-workstation `start_*` wrappers and local-file indexers are not included.

## Reproduction settings

Seeds, sample counts, role constraints and model capacities are embedded in the original modules and the copied scientific protocols. The default Monte Carlo count in the early generic `TrainConfig` is not the 100,000-draw CORE setting: use `r4_final_model`, `eu_core_experiment` and their frozen recipes when reproducing the final paper. Some older names retain `JEPA` or `kernel`; their inclusion provides development traceability and does not change the manuscript's disabled-module conclusions.

For inference comparisons retain the same object order, sample count, seed, and chunk size, since these define the random stream. Paired comparison also requires the same frozen identities and missing-outcome accounting. Changing a ranking or missingness rule creates a new analysis rather than reproducing the existing one.
