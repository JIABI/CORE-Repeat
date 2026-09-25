# Targeted repeat measurements

Code accompanying **Targeted repeat measurements improve the value and cross-site agreement of phenotypic profiles**.

This source folder is refreshed to the final manuscript of 25 September 2026. It includes the exact-geometry consistency tests, the post-hoc EU task-matched measurement comparison, and plotting code with the current Figure 1–7 numbering. Data and fitted artifacts belong in the separate companion folder, `CORE-Repeat-data`.

- Data and frozen-model record: [10.5281/zenodo.22927401](https://doi.org/10.5281/zenodo.22927401).
- Code repository: [JIABI/CORE-Repeat](https://github.com/JIABI/CORE-Repeat).

This repository contains the actual measurement models, direct-prediction comparators, acquisition evaluation, and regression tests used in the study. The Python package retains its development name, `opal2`. CORE is the joint measurement model studied in the manuscript; optional biology and representation modules are experimental comparators, not enabled components of the frozen confirmation policy.

## Start here

There are three different reproduction tasks:

1. **Inspect and reproduce the reported figures/statistics:** use the companion data deposit with `paper/`. This does not require fitting the models again.
2. **Inspect or reuse the algorithms:** install `opal2`, run the tests, then use the array-based APIs listed below.
3. **Refit the complete historical experiments:** use the dated runners in `scripts/`, after acquiring and preparing the upstream datasets and their recorded partitions. The scripts retain the original run names and input-tree conventions. The source-data deposit is not a replacement for all raw images, original feature archives, and every historical intermediate checkpoint.

No data are downloaded and no experiments are started by installation, package import, or `--help`.

## Installation and tests

Python 3.11 or newer is required. The release was tested on Python 3.12.14, CPU, using the package versions recorded in `requirements-tested.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m opal2 --help
```

For an environment matching the recorded numerical versions:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
```

The pinned environment records the versions used; it is not a cross-platform binary lockfile. For exact re-execution, in particular the conditional-quantile HistGB experiment, use the recorded scikit-learn version. Newer versions are not assumed numerically identical. CPU is sufficient. GPU execution is not required for the manuscript results. The tests include small artificial fixtures solely for software validation, not as scientific evidence.

See `docs/RELEASE_VALIDATION.md` for the actual release checks and any skipped data-dependent tests.

## Current CORE components

| Component | Implementation |
|---|---|
| Four-well Gram geometry and exact action utilities | `opal2/gram_geometry.py` |
| Mean fitting: ridge → hierarchical residual → generic reference → STATE50 | `opal2/eu_core_training.py` |
| Conditional joint scatter and amplitude/radial error law | `opal2/eu_core_distribution.py`, `opal2/empirical_radial.py` |
| Frozen confirmation fit, load and inference | `opal2/r4_final_model.py` |
| Ridge, ExtraTrees and HistGB direct comparators | `opal2/eu_r2_direct_baselines.py` |
| Conditional-quantile HistGB distribution and scoring | `opal2/quantile_distribution.py`, `opal2/quantile_direct_evaluation.py` |
| Nonnegative measurement-quantile laws and frozen observable replay | `opal2/observable_quantile_distribution.py`, `opal2/measurement_forecast_replay.py` |
| Identity-aware selection, missing-outcome bounds and resampling | `opal2/r4_evaluation.py`, `opal2/r4_confirmatory_metrics.py` |
| Independent-site evaluation endpoint | `opal2/r4_external_endpoint.py` |
| Joint dependence ablation | `opal2/m4_dependence_ablation.py` |

The `eu_` filenames identify where these APIs were introduced, not a restriction to EU-OPENSCREEN. The same CORE recipe was evaluated on the other development resources. The old `opal2` command-line interface contains early model-development arms and is retained for traceability; its `experiment` command is **not** a one-command implementation of the final paper pipeline. Use the current components and dated runners above.

### Fit the final recipe from prepared development data

```bash
python scripts/fit_r4_model_20260921.py \
  --dataset /path/to/prepared_development \
  --partitions /path/to/frozen_partitions.json \
  --output /path/to/new_final_model
```

This runner implements the paper's exact final EU role counts and protocol; it is not a generic random-split trainer. The model input preparation, assay normalisation, and role counts are specified in `docs/REPRODUCTION.md` and the accompanying protocols.

### Score compatible prepared first-well queries

```bash
python scripts/score_prepared_queries.py \
  --model /path/to/trusted_final_model \
  --query /path/to/prepared_first_wells.npz \
  --population-size 1539 \
  --output /path/to/new_predictions
```

The NPZ accepts `ids`, `groups`, `X`, `chem`, `chem_mask` and optional `layout`, not future outcomes. `X` must use the frozen assay coordinates and normalisation. The frozen FMP model does **not** automatically adapt to another laboratory, plate transform or feature panel. This wrapper uses the original 100,000 draws per object and original three seeds, with both CORE and Gaussian diagnostic scoring; large populations need substantial time and disk space. Saved model objects use PyTorch/joblib serialisation: load only artifacts obtained from a trusted source.

## Data, protocols and reproducibility

The companion data package provides the release data manifest, schemas, provenance, and upstream data-access terms. Large arrays and checkpoints are not committed here. Each upstream dataset keeps its own licensing conditions; code availability does not grant rights to redistribute those datasets.

- `docs/REPRODUCTION.md`: what is required for each reproduction level, inputs, output trees and historical limitations.
- `docs/METHOD_MAP.md`: manuscript evidence to implementation map.
- `protocols/`: scientific protocols and frozen recipes, including explicit development/confirmation distinctions.
- `paper/`: figure and saved-result analysis tools prepared for this release.
- `tests/`: algorithm, data isolation, numerical and policy-evaluation regression tests.

The original confirmation models and acquisition lists were fixed before future outcomes were examined. The EU direct-measurement comparison was specified after confirmation outcomes were opened and is explicitly post hoc; it leaves those original models and lists unchanged. Its primary arm is calibrated direct quantile prediction (CAL), with RAW secondary. The separate development gain-quantile comparison retains RAW as primary and CAL as secondary. Reproducing saved analyses is not a new independent confirmation.

For figures and saved-result checks, set `OPAL2_DATA_ROOT` to the companion `CORE-Repeat-data` directory and follow `paper/README.md`. The paper's current figure/table labels, rather than dated development run names, identify the final evidence.

## Licence and citation

The authors' original software contributions are provided under the MIT licence in `LICENSE`, subject to the material and path exclusions in `LICENSE_NOTES.md`. This grant does not relicense third-party data, annotations, trained artifacts or restricted derivatives. See `THIRD_PARTY_NOTICES.md` for those sources.

Cite the data record [10.5281/zenodo.22927401](https://doi.org/10.5281/zenodo.22927401) and identify the [CORE-Repeat repository](https://github.com/JIABI/CORE-Repeat) revision used. The dataset DOI is not an article DOI. The manuscript title above identifies the associated study.
