"""Independent, read-only numerical audit of completed headroom artifacts.

Writes only INDEPENDENT_AUDIT.json in the new diagnostic run. Saved model
outputs, selected sets, fitting artifacts, and historical runs are not changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .conditional_joint_error_experiment import observable_forward
from .conditional_residual_information import error_targets
from .dual_branch_features import apply_increment
from .empirical_radial import draw_radial, fit_radial, radial_nll, variance_multiplier
from .frozen_acquisition_policy import select_frozen_cohort_plan
from .objective_analysis import fair_crps
from .variance_headroom_experiment import (
    ARMS, SEED, calibrate_rank_map, global_calibration_choice, scalar_grid,
)


PROJECT = Path(__file__).resolve().parents[1]


def _json(path):
    return json.loads(Path(path).read_text())


def _npz(path):
    with np.load(path) as source:
        return {k: source[k].copy() for k in source.files}


def audit(root):
    root = Path(root).resolve()
    if _json(root/'status.json')['state'] != 'COMPLETE':
        raise ValueError('Audit requires all ten completed cells')
    radial = PROJECT/'runs/lincs_empirical_radial_20260916_v1'
    dual = PROJECT/'runs/dual_branch_biology_20260917_v2'
    old = _json(radial/'summary.json')
    manifest = _json(Path(old['reference_run'])/'run_manifest.json')
    prior, original = _npz(radial/'AMP_EMP_LOCAL.npz'), _npz(dual/'CORE.npz')
    support = _npz(dual/'GELU.npz')['resource_support'].astype(bool)
    ids = np.asarray(manifest['ids']); lookup = {v: i for i, v in enumerate(ids)}
    np.testing.assert_array_equal(prior['ids'], ids)
    np.testing.assert_array_equal(original['ids'], ids)
    records = {r['fold']: r for r in manifest['folds']}
    aggregates = {arm: _npz(root/(arm+'.npz')) for arm in ARMS}
    for values in aggregates.values():
        np.testing.assert_array_equal(values['ids'], ids)
        np.testing.assert_array_equal(values['support'], support)
        np.testing.assert_array_equal(values['actual'], original['actual'])
        for value in values.values():
            if np.issubdtype(value.dtype, np.number):
                assert np.isfinite(value).all()
    cells = []; empirical_lower = []; guard_lower = []; guard_underflow_z = []
    seen = np.zeros(len(ids), int)
    prefix_max_error = 0.
    for number, cell in enumerate(old['cells']):
        fold, half = cell['fold'], cell['half']
        directory = root/f'cell_{fold}_{half}'
        assert _json(directory/'complete.json')['state'] == 'COMPLETE'
        q = np.asarray([lookup[v] for v in cell['query_ids']])
        cal = np.asarray([lookup[v] for v in cell['representative_ids']])
        fit = np.asarray(records[fold]['fit'])
        np.add.at(seen, q, 1)
        ref = _npz(radial/f'cell_{fold}_{half}_radial.npz')
        search = _npz(directory/'search.npz'); calibration = _json(directory/'calibration.json')
        np.testing.assert_array_equal(ref['cal_ids'], ids[cal])
        np.testing.assert_array_equal(ref['query_ids'], ids[q])
        np.testing.assert_array_equal(search['cal_ids'], ids[cal])
        np.testing.assert_array_equal(search['ids'], ids[q])
        assert not set(ids[fit]) & set(ids[np.r_[cal, q]])
        assert not set(calibration['calibration_groups']) & set(calibration['query_groups'])
        stats = _json(Path(old['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json')
        scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
        law = fit_radial(ref['amplitude_radii'])
        multiplier = variance_multiplier(law, ref['local_weights'])
        np.testing.assert_allclose(prior['scatter_u'][q]*multiplier[:, None, None],
                                   original['covariance_u'][q], rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(prior['mean_u'][q], original['mean_u'][q], rtol=0, atol=0)
        np.testing.assert_allclose(observable_forward(prior['actual_u'][q]*scale+center)[0],
                                   original['actual'][q], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(ref['cal_residual'], prior['actual_u'][cal]-prior['mean_u'][cal],
                                   rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(np.linalg.norm(np.linalg.solve(np.linalg.cholesky(ref['cal_amp_scatter']),
            ref['cal_residual'][..., None])[..., 0], axis=1), ref['amplitude_radii'], rtol=2e-13, atol=2e-13)
        condition = joblib.load(PROJECT/f'runs/conditional_residual_information_20260916_v1/fold_{fold}/conditioners.joblib')
        for obj in (condition['transformer'], condition['amplitude'], condition['conditional']['DESCRIPTORS']):
            assert set(obj.fit_ids) == set(ids[fit])
            assert not set(obj.fit_ids) & set(ids[np.r_[cal, q]])
        _, global_info = global_calibration_choice(search['cal_scores'],
            np.asarray(calibration['calibration_groups']), scalar_grid())
        assert global_info == calibration['global_choice']
        mapped = {}
        for arm, suffix in [('AMPLITUDE_RANK_CAL', 'amplitude'), ('LEARNED_RANK_CAL', 'learned'),
                            ('TRUE_ENERGY_RANK_CAL', 'rank6_energy')]:
            values, info = calibrate_rank_map(search['cal_'+suffix], search['query_'+suffix],
                                              search['cal_optimal_scalar'])
            assert info == calibration['rank_mappings'][arm]
            mapped[arm] = np.column_stack((values, values)); mapped[arm][~support[q]] = 0.
        ridge = _npz(Path(manifest['frozen_base_reference_run'])/'folds'/f'fold_{fold}'/'ridge.npz')['covariance']
        take = np.r_[cal, q]
        ridge_rows = np.broadcast_to(ridge, (len(take), 9, 9)) if ridge.shape == (9, 9) else ridge[take]
        historical = error_targets(prior['mean_u'][take]*scale+center,
            ridge_rows*scale[None, :, None]*scale[None, None, :],
            (prior['actual_u'][take]-prior['mean_u'][take])*scale)
        np.testing.assert_array_equal(search['cal_rank6_energy'], historical[:len(cal), 1])
        np.testing.assert_array_equal(search['query_rank6_energy'], historical[len(cal):, 1])
        core = _npz(directory/'CORE.npz')
        for arm in ARMS:
            out = _npz(directory/(arm+'.npz'))
            np.testing.assert_array_equal(out['ids'], ids[q])
            for key, value in out.items():
                if key != 'ids':
                    np.testing.assert_array_equal(aggregates[arm][key][q], value)
            eta = out['increment']
            assert np.max(np.abs(eta)) <= np.log(4.)+1e-12
            np.testing.assert_array_equal(eta[~support[q]], np.zeros((np.sum(~support[q]), 2)))
            for key in ('predicted', 'p_null', 'crps', 'brier', 'nll', 'joint_coverage_by_level'):
                np.testing.assert_array_equal(out[key][~support[q]], core[key][~support[q]])
            cov = apply_increment(prior['mean_u'][q]*scale+center, scale, prior['scatter_u'][q], eta)
            np.testing.assert_allclose(out['nll'], radial_nll(prior['actual_u'][q]-prior['mean_u'][q],
                cov, law, ref['local_weights']), rtol=2e-12, atol=2e-12)
            variance = out['gamma_mc_se']**2+.04*out['null_mc_se']**2-.4*out['mean_null_mc_covariance']
            np.testing.assert_allclose(out['score_mc_se']**2, variance, rtol=1e-10, atol=2e-17)
            paired_se = (out['crps_mc_blocks']-core['crps_mc_blocks']).std(1, ddof=1)/np.sqrt(20)
            np.testing.assert_array_equal(paired_se, out['crps_paired_mc_se'])
            choice = select_frozen_cohort_plan(ids[q], out['predicted'], out['p_null'], cell['budget'])
            np.testing.assert_array_equal(out['selected'], np.asarray(choice.selected_mask, int))
            if arm in mapped:
                np.testing.assert_array_equal(eta, mapped[arm])
            if arm in ('H1_CRPS', 'H2_CRPS', 'H1_NLL', 'H2_NLL'):
                key = dict(H1_CRPS='eta_scalar_crps', H2_CRPS='eta_two_crps',
                           H1_NLL='eta_scalar_nll', H2_NLL='eta_two_nll')[arm]
                expected = search[key].copy(); expected[~support[q]] = 0.
                np.testing.assert_array_equal(eta, expected)
            # Replay one independent MC block using original observable_forward,
            # not the runner's optimized endpoint implementation.
            if number == 0:
                rng = np.random.default_rng(SEED+9000000)
                normal = rng.normal(size=(5000, 4, 9))
                rr = np.random.default_rng(SEED+9000000+47000)
                mix = rr.random((100000, 4))[:5000]; kernel = rr.random((5000, 4))
                error = draw_radial(law, ref['local_weights'][:4], cov[:4], normal, mix, kernel)
                gamma = observable_forward((prior['mean_u'][q[:4]][None]+error)*scale+center)[0]
                score = fair_crps(gamma, original['actual'][q[:4]])
                np.testing.assert_allclose(score, out['crps_mc_blocks'][:4, 0], rtol=3e-14, atol=3e-16)
                prefix_max_error = max(prefix_max_error, float(np.max(np.abs(score-out['crps_mc_blocks'][:4, 0]))))
        raw_mean = prior['mean_u'][q]*scale+center
        max_sd = 2*np.sqrt(np.diagonal(prior['scatter_u'][q], axis1=1, axis2=2))*scale
        rmax = np.exp(np.max(law['log_centers'])+law['bandwidth'])
        empirical_lower.append(float(np.min((raw_mean-max_sd*rmax)[:, [3, 5, 8]])))
        guard_lower.append(float(np.min((raw_mean-max_sd*12)[:, [3, 5, 8]])))
        log_underflow = np.log(np.nextafter(0., 1.))/2
        guard_underflow_z.append(float(np.min(((raw_mean-log_underflow)/max_sd)[:, [3, 5, 8]])))
        cells.append(dict(fold=fold, half=half, queries=len(q), calibration=len(cal),
                          all_nine_arms_replayed=True, frozen_core_distribution_replayed=True,
                          calibration_map_replayed=True, historical_rank6_frame_replayed=True))
    np.testing.assert_array_equal(seen, np.ones(len(ids), int))
    search_seeds = ({SEED+100000*i+50000 for i in range(10)}
                    | {SEED+100000*i+1009*j for i in range(10) for j in range(40)})
    final_seeds = {SEED+9000000+100000*i+10007*j+k for i in range(10)
                   for j in range(0, len(old['cells'][i]['query_ids']), 4) for k in (0, 47000)}
    assert not search_seeds & final_seeds
    summary = _json(root/'summary.json')
    for arm in ('TRUE_ENERGY_RANK_CAL', 'H1_CRPS', 'H2_CRPS', 'H1_NLL', 'H2_NLL'):
        assert summary['metrics'][arm]['hindsight'] is True
    result = dict(state='PASS', audit='independent artifact and formula replay', objects=len(ids), cells=cells,
        optimization_final_seed_collisions=0, original_forward_prefix_max_error=prefix_max_error,
        unchanged_mean_and_core_radial_law=True, unsupported_increments_exact_zero=True,
        aggregate_cell_fields_exactly_match=True, all_numeric_aggregate_fields_finite=True,
        cal_maps_use_only_saved_disjoint_calibration_scores=True,
        query_outcome_usage='Only explicitly labelled hindsight arms and metric scoring',
        historical_rank6_frame='original RIDGE covariance, not current CORE total energy',
        monte_carlo_score_variance_identity_replayed=True,
        paired_crps_block_standard_errors_replayed=True,
        empirical_support_min_log_diagonal=min(empirical_lower),
        gaussian_guard_12sigma_min_log_diagonal=min(guard_lower),
        gaussian_guard_min_sd_to_squared_underflow=min(guard_underflow_z),
        gaussian_guard_underflow_union_log_probability_upper=float(
            norm.logsf(min(guard_underflow_z))+np.log(len(ids)*len(ARMS)*100000*3)),
        numerical_boundary_note='Original evaluator checks squared diagonal underflow; fast runner lacks that extra guard. '
            'For this run every bounded-eta empirical draw has log diagonal >= reported support minimum; '
            'even 12-SD Gaussian guard excursions stay far above the approximately -372 underflow threshold.',
        restrictions='No continuous or learnability bound; hindsight coverage/policy not deployable; '
            '20-block integration error is distinct from finite-object sampling uncertainty.',
        mutation_scope='Only this new INDEPENDENT_AUDIT.json; no saved predictions or old runs changed')
    write_json(root/'INDEPENDENT_AUDIT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = audit(args.output)
    print(result['state'], len(result['cells']), result['objects'])
