# Reproduce publication displays and saved-result analyses

This directory uses the accompanying data release, not the author's workstation or private skill installation. It does not download datasets or fit models.

## Setup

Install the repository and its plotting dependencies:

```bash
python -m pip install -e '.[test]'
python -m pip install matplotlib Pillow PyMuPDF requests
export OPAL2_DATA_ROOT=/absolute/path/to/zenodo_data
```

NumPy, pandas and RDKit are already package dependencies. Arial was used for the manuscript; DejaVu Sans is the portable fallback when Arial is unavailable. Exact typography can depend on installed fonts; data and plotting coordinates are retained.

## Render figures

```bash
python paper/reproduce_figures.py
```

Or specify main figure numbers, for example `python paper/reproduce_figures.py 2 5 6`. Supplementary Figures 1, 2 and 4 are generated alongside main Figures 1, 2 and 5; selectors `S3` and `S5` generate the remaining two. Outputs default to `paper/figures/` and `paper/qa/`; override with `OPAL2_FIGURE_OUT` and `OPAL2_QA_OUT`.

Historical filenames are retained: main Figure 3 is `fig7_amplitude_dependence`, Figure 4 is `fig6_measurement_coverage`, Figure 6 is `fig3_optional_information`, and Figure 7 is `fig4_confirmation`. Use the data release's `FIGURE_SOURCE_INDEX.csv` for the exact mapping, not these historical numbers.

The released figure scripts read saved source CSV/NPZ values. The RxRx3 pair and response panels use exact plot-only exports rather than reopening the original 35 training units. No model fit, acquisition reranking or new scientific resampling occurs during plotting. Deterministic panel summaries are exported to `paper/qa/`, not written over the downloaded source data. Input measurements and saved selections are not changed.

## Reanalyse existing lists

```bash
python paper/scripts/cost_sensitivity_fixed_lists.py --help
python paper/scripts/confirmation_rank_gradient.py --help
python paper/scripts/v10_r4_random_supplement.py --help
python paper/scripts/plate_structure_and_verifier_rotation.py --check
```

The cost and rank programs accept explicit input/output paths. The random-comparison program defaults to the released research tree, reproduces the original 10,000-resample seed and checks the original CORE intervals before adding HistGB. It writes to `paper/qa/r4_random_reproduction/`, leaving the frozen outcomes and selections unchanged. It is slower than figure rendering, but requires no training or new predictive Monte Carlo draws.

`prepare_microscopy.py` reconstructs the public JUMP illustration and may contact the upstream metadata URLs; the complete cached originals and processed crops already included in the data release are sufficient for normal figure rendering.

The role-rotation check recomputes Supplementary Table 2 from the four released measurement arrays. It compares every saved cell and does not treat the reused roles as independent remeasurements.

## Scientific checks

Run `python paper/verify_release_data.py` to check the main population counts, fixed budgets, aligned identities and missing-outcome ledger. The 58 final displayed tables are archived separately from their full-precision source data; their text cells preserve the reported formatting rather than silently creating additional precision.
