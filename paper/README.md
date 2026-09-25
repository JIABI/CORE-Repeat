# Reproduce publication displays and saved-result analyses

This directory uses the accompanying data release, not the author's workstation or private skill installation. It does not download datasets or fit models.

## Setup

Install the repository and its plotting dependencies:

```bash
python -m pip install -e '.[test,figures]'
export OPAL2_DATA_ROOT=/absolute/path/to/CORE-Repeat-data
```

NumPy, pandas and RDKit are already package dependencies. Arial was used for the manuscript; DejaVu Sans is the portable fallback when Arial is unavailable. Exact typography can depend on installed fonts; data and plotting coordinates are retained.

## Render figures

```bash
python paper/reproduce_figures.py
```

Or specify current manuscript figure numbers, for example `python paper/reproduce_figures.py 1 2 3`. Supplementary Figures 1, 2 and 4 are generated alongside main Figures 1, 3 and 6; selectors `S3` and `S5` generate the remaining two. Outputs default to `paper/figures/` and `paper/qa/`; override with `OPAL2_FIGURE_OUT` and `OPAL2_QA_OUT`.

Historical filenames are retained; the driver uses the final manuscript numbering:

| Main figure | Saved filename stem | Content |
|---|---|---|
| 1 | `fig1_measurement_structure` | Observed repeat value and joint-geometry representation |
| 2 | `fig4_confirmation` | Frozen selection and external agreement |
| 3 | `fig2_prediction_decision` | Budget curves and rank profiles |
| 4 | `fig7_amplitude_dependence` | Amplitude controls and dependence ablation |
| 5 | `fig6_measurement_coverage` | Measurement-observable coverage |
| 6 | `fig5_risk_cost` | Selected-set risk and resource costs |
| 7 | `fig3_optional_information` | Biological relationships and response prediction |

The companion `FIGURE_SOURCE_INDEX.csv` records the same mapping. Main Table 3 is the post-hoc EU measurement comparison; its saved full-precision results are in `source_data/eu_direct_measurement_20260924/`.

The released figure scripts read saved source CSV/NPZ values. The RxRx3 pair and response panels use exact plot-only exports rather than reopening the original 35 training units. No model fit, acquisition reranking or new scientific resampling occurs during plotting. Deterministic panel summaries are exported to `paper/qa/`, not written over the downloaded source data. Input measurements and saved selections are not changed.

## Reanalyse existing lists

```bash
python paper/scripts/cost_sensitivity_fixed_lists.py --help
python paper/scripts/confirmation_rank_gradient.py --help
python paper/scripts/v10_r4_random_supplement.py --help
python paper/scripts/plate_structure_and_verifier_rotation.py --check
python paper/scripts/reproduce_null_types.py
```

The cost and rank programs accept explicit input/output paths. The random-comparison program defaults to the released research tree, reproduces the original 10,000-resample seed and checks the original CORE intervals before adding HistGB. It writes to `paper/qa/r4_random_reproduction/`, leaving the frozen outcomes and selections unchanged. It is slower than figure rendering, but requires no training or new predictive Monte Carlo draws.

`prepare_microscopy.py` reconstructs the public JUMP illustration and may contact upstream metadata URLs; it additionally requires `requests`. The cached originals and processed crops in the companion data are sufficient for normal offline figure rendering.

The role-rotation check recomputes Supplementary Table 2 from the four released measurement arrays. It compares every saved cell and does not treat the reused roles as independent remeasurements.

The NULL-type reproduction uses the unchanged frozen confirmation selections and existing measurements. Its summary, selected-outcome ledger and audit are written to `paper/qa/null_types_20260925/`; the downloaded source files remain unchanged.

## Scientific checks

Run `python paper/verify_release_data.py` to check the main population counts, fixed budgets, aligned identities, missing-outcome ledger and EU direct-measurement summaries. The 64 numbered tables (3 main and 61 supplementary) are archived separately from their full-precision source data; continued panels share their table number and export. Displayed text cells preserve the manuscript formatting.
