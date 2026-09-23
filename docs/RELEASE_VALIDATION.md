# Release validation

Validation completed for the 24 September 2026 staging release. These checks apply to the code, not to a new scientific experiment.

## Executed checks

| Check | Result |
|---|---|
| Complete repository regression suite | **1,546 passed, 8 skipped, 12 subtests passed**; no failures |
| Suite duration in the tested environment | 119.18 seconds |
| Build an installable wheel with no dependency changes | Passed; `opal2_measurement_model-0.2.0-py3-none-any.whl` |
| Install that wheel into a separate target and import from outside the source tree | Passed |
| Import current mean, distribution, confirmation and quantile APIs from installed wheel | Passed |
| Historical CLI help and both current fit/score wrapper help commands | Passed |
| Load the actual saved final CORE/HistGB/distribution state using the release code | Passed; 904 development identities, CORE in evaluation mode |
| Original core algorithm preservation | All original Python ASTs match after removing only the recorded protocol-path and version-string packaging edits |
| Author-machine absolute paths or credential-pattern scan in distributed core/scripts/tests/docs/protocols | No matches |
| Documented Python file references | All explicit file references resolve |

The numerical test environment was Python 3.12.14 on macOS arm64, CPU, using `requirements-tested.txt`. The full test command was:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest -q -rs
```

## Skipped tests

Eight pre-existing tests require optional local historical data/checkpoints and were skipped in the clean source folder:

- One dual-branch saved-cell isolation test.
- Five saved STATE50 transfer checks.
- One JUMP original-development metadata check.
- One explicit `OPAL2_LEGACY_ROOT` real-archive integration test.

The ordinary numerical, schema, fitting-isolation, selection, missingness, covariance, calibration and resampling tests ran. Skips were not converted into passes. The separate real final-model load check succeeded without copying those historical test inputs into the code repository.

## Packaging changes

- Historical scientific protocol files are organised under `protocols/historical/`; quoted paths in callers and one corresponding test fixture were updated. Their scientific contents and algorithms were not changed.
- The package's internal version string now agrees with the pre-existing project version, 0.2.0.
- `python -m opal2` forwards to the existing CLI.
- `scripts/score_prepared_queries.py` is a new thin, tested wrapper around the original frozen inference function. It does not alter model settings, sample counts, seeds, action costs or policy ranking.
- Joblib and threadpoolctl are explicit direct dependencies; optional figure dependencies are separated in project metadata.
- Author-workstation background launch wrappers and local-file indexing scripts were excluded. Archived scientific orchestration modules remain available, with their platform/input requirements documented.

The release check did not rerun all biological training, Monte Carlo predictions, or create a new confirmation cohort. Historical background modes were not tested on Linux or Windows. The companion data and `paper/` tools support saved-result reproduction separately from full model fitting.

## Paper reproduction checks

- All 7 main figures and 5 supplementary figures rendered from the extracted-data layout using the portable figure driver. The release scripts do not require the authors' workstation paths or private figure skills. The rendered-canvas checks reported no text outside the drawn canvas.
- The two main and 56 supplementary tables have released cell exports and complete source indices. All listed data and code paths resolve.
- Recomputing the four-role gain from 13,141 development observations and 1,520 observed confirmation outcomes agrees with the saved publication values to below 1.5e-15 maximum absolute error. The 1,539 / 1,527 / 1,520 confirmation ledger and 192 selections per policy were checked.
- The portable role-rotation diagnostic reproduces all four resources' saved Supplementary Table 2 source cells, with zero maximum numerical difference in the tested environment.
- Original manuscript figure PDF/SVG files are retained in the data package as reference renders. Typography in new renders may depend on locally available fonts.

The small canvas helper excludes undrawn out-of-range tick artists from its warnings. That change affects visual QA only, not plotted points, axis limits or numerical results.
