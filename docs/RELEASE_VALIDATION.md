# Release validation

The 25 September 2026 source-folder refresh was checked offline against the final companion `CORE-Repeat-data` folder. These checks validate packaging and saved-result reproduction; they do not constitute a new scientific experiment.

## Checks executed for this refresh

| Check | Result |
|---|---|
| Focused numerical and wrapper regression tests | 75 passed in 4.22 seconds; no failures or skips |
| Python syntax | All 489 Python files parsed successfully |
| Scientific implementation preservation | Core-module ASTs match the canonical analysis, allowing only the pre-existing historical-protocol path relocation and version-string correction |
| Newly included EU measurement modules, runner and tests | Byte-identical to the canonical analysis sources |
| EU measurement runner, final fit wrapper, prepared-query score wrapper and figure driver help | Passed |
| Confirmation and development saved-data verification | Passed for 13,141 development observations and 1,520 complete confirmation outcomes |
| Confirmation ledger and frozen selections | 1,539 qualified; 1,527 eligible first profiles; 1,520 observed gains; 192 selections per policy |
| EU task-matched measurement comparison | Saved CAL-primary paired comparisons and 21.8% nominal-95%-interval width reduction verified; unresolved CRPS contrast retained |
| Final table index | 3 main plus 61 supplementary numbered tables, including main Table 3 |
| Figure reproduction | All 7 main and 5 supplementary figures rendered from packaged saved inputs; scripts reported no out-of-canvas text issues |
| NULL-type reproduction | Both exported CSVs are byte-identical to the companion source files |
| Four-resource role-rotation table | Reproduced 13,141 observations; maximum cell difference from saved table 1.1103 × 10⁻¹⁶ |
| Package contents | No datasets, fitted weights, generated figures, caches or raw logs in the code folder |
| Author-machine paths and credential-pattern scan | No matches in distributed code or documentation |

The maximum absolute difference between gains recomputed from released measurements and saved publication gains was **1.4433 × 10⁻¹⁵**. The NULL-type helper reproduced the unchanged frozen selections; it did not fit models or select new objects.

The focused numerical tests used the existing Python 3.12.14 macOS arm64 CPU environment recorded in `requirements-tested.txt`:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m pytest -q -rs -p no:cacheprovider \
  tests/test_gram_geometry.py \
  tests/test_measurement_forecast_replay.py \
  tests/test_observable_quantile_distribution.py \
  tests/test_quantile_distribution.py \
  tests/test_quantile_direct_evaluation.py \
  tests/test_m4_dependence_ablation.py \
  tests/test_r4_evaluation.py \
  tests/test_r4_confirmatory_metrics.py \
  tests/test_release_wrappers.py
```

Rendering used the existing Python 3.12.0 environment with matplotlib 3.10.9, NumPy 2.1.3, pandas 3.0.3, SciPy 1.17.1, Pillow 12.2.0 and RDKit 2026.03.5. No environment was installed or enlarged for this refresh. Figure dependencies are optional (`pip install -e '.[figures]'`); the numerical-only environment does not itself contain matplotlib. Typography can vary with local fonts. Generated validation outputs were kept outside the code folder.

## Scope and remaining requirements

- This refresh reused the prior staged package and ported current paper scripts through its existing packaging utility. Paper-script changes are restricted to input/output paths, pre-exported saved plot inputs and local canvas checks. Current manuscript Figure 1–7 selectors are documented in `paper/README.md`.
- The full repository regression suite, wheel-build/install check and every historical training pipeline were **not rerun** for this refresh. The focused tests above cover geometry, the added EU measurement comparison, dependence ablation, confirmation statistics and release wrappers.
- Exact EU measurement re-execution uses scikit-learn 1.9.1, the recorded prepared/frozen inputs, and a separate writable analysis tree; see `docs/REPRODUCTION.md`. It is not part of the quick saved-result checks.
- Full historical reproduction still requires the upstream raw archives and recorded inputs not included in the companion release. Historical POSIX/background modes have not been validated on Windows or retested on Linux.
- This local code folder is not a remote publication. Original software is MIT-licensed only within the scope of `LICENSE_NOTES.md`; third-party notices and reserved source-dependent paths retain their stated terms. The mixed-rights companion data package requires a separate author licensing decision before public redistribution.

## Historical validation: 24 September staging release

The preceding staging release recorded **1,546 passed, 8 skipped and 12 subtests passed** in 119.18 seconds under Python 3.12.14/macOS arm64. It also passed a no-dependency-change wheel build, installation into a separate target, installed API imports, CLI/wrapper help and loading the actual saved final CORE/HistGB/distribution state for 904 development identities. These are retained historical results, not claims that those checks were repeated on 25 September.

Its eight skipped tests required optional historical inputs: one dual-branch saved-cell isolation test, five saved STATE50 transfer checks, one JUMP development-metadata check and one explicit `OPAL2_LEGACY_ROOT` archive integration test. They were not counted as passes.

That release rendered all seven main and five supplementary figures, reproduced the four-resource role-rotation table with zero numerical difference, and verified the same confirmation ledger and saved gains. Its table export covered the then-current two main and 56 supplementary tables; that historical table count is superseded by the **three main and 61 supplementary tables** verified above.

Packaging changes inherited from that release include the `protocols/historical/` relocation, version 0.2.0 alignment, the `python -m opal2` entry point, the tested prepared-query scoring wrapper, explicit joblib/threadpoolctl dependencies, separated optional plotting dependencies and exclusion of author-workstation launch/indexing helpers. Original scientific algorithms, settings, sample counts, seeds, action costs and policy rankings were retained.
