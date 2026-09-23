# Evidence-to-code map

This map distinguishes the final measurement model from optional development experiments. Figures and Supplementary Notes in the manuscript define the scientific comparisons; file dates preserve execution history.

| Evidence family | Core calculation | Checks |
|---|---|---|
| Repeated-profile geometry and realised gain | `gram_geometry.py`, `replicate_diagnostics.py`, `diagnostics.py` | `test_gram_geometry.py`, `test_replicate_diagnostics.py`, `test_observed_diagnostics_r2.py` |
| CORE conditional mean | `eu_core_training.py`, `hierarchical_stability_ridge.py`, `state_biology_kernel.py` | `test_eu_core_training.py`, `test_state_biology_kernel.py` |
| Conditional joint error and empirical radius | `conditional_joint_error.py`, `eu_core_distribution.py`, `empirical_radial.py` | `test_conditional_joint_error.py`, `test_eu_core_distribution.py`, `test_empirical_radial.py` |
| Direct prediction and quantile control | `eu_r2_direct_baselines.py`, `quantile_distribution.py`, `quantile_direct_evaluation.py` | `test_eu_r2_direct_baselines.py`, `test_quantile_distribution.py`, `test_quantile_direct_evaluation.py` |
| Measurement-dependence ablation | `m4_dependence_ablation.py`, `m4_dependence_summary.py` | `test_m4_dependence_ablation.py` |
| Optional reference-summary/representation experiments | `dual_branch_biology.py`, `support_gated_biology.py`, `conditional_state_representation.py` | Corresponding `test_*.py` files; unsupported/off-state invariants |
| Cross-dose association and response-only comparisons | `crossdose_response.py`, `relation_variance_components.py`, dated cross-dose runners | Corresponding tests and the saved-output analysis in the companion data |
| Hindsight scale and independent model-generated realisation | `variance_headroom_math.py`, `null_scale_oracle_experiment.py`, `direct_risk_scale_experiment.py` | `test_variance_headroom_math.py`, `test_null_scale_oracle_experiment.py`, `test_direct_risk_scale.py` |
| Frozen fit and X-only prediction | `r4_final_model.py` | `test_r4_final_model.py`, `test_release_wrappers.py` |
| Missing-outcome bounds and paired resampling | `r4_confirmatory_metrics.py`, `r4_evaluation.py`, `r4_primary_analysis.py` | `test_r4_evaluation.py`, `test_r4_primary_analysis.py`, `test_r4_fdp_missing_resampling.py` |
| Cross-site morphology endpoint | `r4_external_endpoint.py` | `test_r4_external_endpoint.py` |
| Amplitude-only controls | `m3_amplitude_controls.py` | `test_m3_amplitude_controls.py` |

Paths in the calculation column are under `opal2/` unless stated otherwise; test paths are under `tests/`. The final statistical supplement comparing both frozen policies with random allocation is also included among `paper/` tools. It keeps a within-resample same-budget random comparator and is distinct from reranking the frozen original cohort.

Unit tests use synthetic numerical fixtures to test algebra, numerical implementation, access boundaries and invariance. They do not replace the actual biological data in the paper. Results reproduction uses the separate released source-data and research artifacts.
