"""Complete original full-space L on JUMP R2's fixed TRAIN and query identities.

Only this new run is written. Existing CORE fits, predictions and controls are
read-only. Object-block checkpoints make the expensive 100,000 full-profile
draws resumable without changing L or replacing it with a geometry surrogate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_r2_core_comparison_20260917 as comparison
from opal2.biology_kernel_evaluation import write_json
from opal2.closed_form_baseline import ClosedFormBaseline, fit_baseline
from opal2.gram_reference import ClosedFormGramReference
from opal2.gram_geometry import gram_to_coordinates, gram_gains
from opal2.conditional_joint_error_experiment import observable_forward, interval_scores
from opal2.empirical_radial_experiment import interval_levels, LEVELS
from opal2.objective_analysis import fair_crps
from opal2.reference_information_diagnostic import bootstrap_difference

SOURCE = PROJECT / 'runs/jump_r2_completion_20260918_v1'
DATA = PROJECT / 'data/source5_primary_fullcontrols/measurements.npz'
ROOT = PROJECT / 'runs/jump_matched_l_20260918_v1'
REPORT = PROJECT / 'reports/jump_matched_l_20260918_v1'
SEED, SAMPLES, OBJECT_CHUNK, DRAW_CHUNK = 20260918, 100000, 2, 512
L_FIT = dict(k=200, clip=8., noise_shrinkage=.05, variance_floor=1e-6)
ROLE_KEYS = ('TRAIN', 'VALIDATION', 'REF_FIT', 'DIST_CAL', 'DEV_EVAL')
ARM = 'L_TRAIN_MATCHED_FULL'


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def resolve_parts(ids, manifest):
    """Read exact existing ordered roles; no fresh data-dependent assignment."""
    lookup = {str(oid): i for i, oid in enumerate(ids)}
    if len(lookup) != len(ids):
        raise ValueError('Unique object identities required')
    parts, seen = [], np.zeros(len(ids), int)
    for record in manifest['parts']:
        part = {key: np.asarray([lookup[oid] for oid in record[key]], int) for key in ROLE_KEYS}
        joined = np.concatenate(list(part.values()))
        if not np.array_equal(np.sort(joined), np.arange(len(ids))):
            raise ValueError('Roles must partition the opened population exactly')
        seen[part['DEV_EVAL']] += 1
        parts.append(part)
    if len(parts) != 5 or not np.all(seen == 1):
        raise ValueError('Each object must be queried exactly once')
    return parts


def load_inputs():
    manifest = json.loads((SOURCE / 'run_manifest.json').read_text())
    if manifest['n'] != 639 or manifest['samples'] != SAMPLES or any(
            manifest[key] for key in ('final_opened', 'fifth_repeat_opened', 'confirmation_opened')):
        raise ValueError('Unexpected JUMP scope or precision')
    old = json.loads(Path(manifest['old_manifest']).read_text())
    if old['config']['l_fit'] != L_FIT:
        raise ValueError('Original L fitting recipe changed')
    data = comparison.read_npz(SOURCE / 'identity_layout.npz')
    core = comparison.read_npz(SOURCE / 'CORE_ORIGINAL.npz')
    observed = comparison.read_npz(SOURCE / 'observables.npz')
    with np.load(DATA, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['ids'], data['ids'])
        data['Y'] = saved['Y'].copy()
    if data['Y'].shape != (639, 4, 3617) or not np.isfinite(data['Y']).all():
        raise ValueError('Unexpected full original endpoint measurements')
    for other in (core, observed):
        np.testing.assert_array_equal(other['ids'], data['ids'])
    np.testing.assert_array_equal(core['actual'], observed['actual'])
    parts = resolve_parts(data['ids'], manifest)
    for f, part in enumerate(parts):
        np.testing.assert_array_equal(part['DEV_EVAL'], np.flatnonzero(core['fold'] == f))
    return data, core, observed, parts, manifest


def resource_counts(part):
    n = {key: len(part[key]) for key in ROLE_KEYS}
    available = 4 * sum(n[key] for key in ROLE_KEYS[:-1])
    return dict(role_objects=n, common_available_label_wells=available,
        L_used_label_wells=4*n['TRAIN'], CORE_used_or_selected_label_wells=available,
        L_unused_label_wells=4*sum(n[key] for key in ROLE_KEYS[1:-1]),
        L_used_roles=['TRAIN'], L_unused_roles=list(ROLE_KEYS[1:-1]),
        L_reference_new_wells=0, CORE_reference_new_wells=4*n['REF_FIT'],
        CORE_reference_X_already_available_new_wells=3*n['REF_FIT'],
        L_fully_new_fit_well_cost=.01*4*n['TRAIN'],
        CORE_fully_new_fit_well_cost=.01*available,
        query_initial_wells=n['DEV_EVAL'], query_replay_future_wells=3*n['DEV_EVAL'],
        initial_and_replay_costs_in_primary_gamma=False,
        control_cost='Shared inherited endpoint preprocessing; not recharged per fold')


def freeze_plan(parts, source_manifest):
    ROOT.mkdir(exist_ok=True, parents=True)
    REPORT.mkdir(exist_ok=True, parents=True)
    plan = dict(arm=ARM, source=str(SOURCE), data=str(DATA), samples=SAMPLES,
        seed=SEED, object_chunk_size=OBJECT_CHUNK, draw_chunk_size=DRAW_CHUNK,
        cpu_threads=1, fit=L_FIT, parts=source_manifest['parts'],
        fitting_roles=['TRAIN'], unused_roles=['VALIDATION', 'REF_FIT', 'DIST_CAL'],
        match='identical folds, TRAIN identities, query identities, endpoint, MC count and available label-resource ceiling; actual used labels differ',
        fixed_hyperparameters=True, mean_and_covariance_from_same_original_joint_L=True,
        original_L_changed=False, existing_31_arms_refitted=False,
        endpoint_dimensions=3617, all_endpoint_residual_dimensions_sampled=True,
        block_seed='20260918 + 100*fold + seed_offset + 1000000*block_start',
        common_random_numbers='Same nominal fold seeds as CORE. Full-profile L and 9D CORE have different Gaussian dimensions and RNG consumption, so individual draws are not paired common random numbers.',
        primary_seed_offset=0, optional_sensitivity_seed_offsets=[100000, 200000],
        available_geometry_scores=['MC mean MSE', 'energy', 'coordinate coverage', 'observable CRPS/coverage', 'Gamma CRPS/coverage'],
        unavailable_geometry_scores=['analytic 9D NLL', 'analytic joint-geometry coverage'],
        profile_nll='Exact full original-coordinate 3-future-profile NLL; separate diagnostic, never compared with CORE 9D NLL',
        calibration='Fixed prediction diagnostics only; no new CAL fitting or correction',
        cost_per_new_well=.01, main_lambda=.2, control_lambda=0.,
        budget='Within each existing query fold: floor(.25*N) wells, k=budget//2; object-ID ascending ties',
        final_opened=False, fifth_repeat_opened=False, confirmation_opened=False,
        formal_certificate=False, resources=[resource_counts(part) for part in parts])
    # Existing identical manifests are immutable, including during concurrent
    # post-fit MC runs. Primary execution creates both before starting fits.
    for path in (ROOT / 'run_manifest.json', REPORT / 'run_manifest.json'):
        if path.exists():
            if json.loads(path.read_text()) != plan:
                raise ValueError('Existing run has a different frozen plan')
        else:
            write_json(path, plan)
    return plan


def fit_original_l(y_train, train_ids, seed):
    """No reference/calibration/query outcomes are accepted by this fitter."""
    model = fit_baseline(y_train, random_state=seed, **L_FIT)
    model.metadata.update(train_ids=np.asarray(train_ids).tolist(),
        role='exact existing TRAIN only', recipe='unchanged complete original L')
    return model


def sample_original_l(model, x, *, samples, seed, draw_chunk=DRAW_CHUNK):
    """Invoke the existing complete profile sampler before reducing to Gram."""
    result = ClosedFormGramReference(model).sample(x, samples, seed=seed,
        object_chunk_size=OBJECT_CHUNK, draw_chunk_size=draw_chunk)
    return result.grams, result.mean_profile_gram


def score_draws(grams, actual_raw, actual, stats, norm2):
    """All comparable scores use the same full-profile samples and CORE units."""
    samples, n = grams.shape[:2]
    if samples < 4 or samples % 2:
        raise ValueError('Even sample count >=4 required for energy score')
    raw = gram_to_coordinates(torch.as_tensor(grams)).numpy()
    gamma, obs, difference, cosine = observable_forward(raw)
    np.testing.assert_allclose(gamma, gram_gains(torch.as_tensor(grams)).numpy()[..., 2], atol=1e-11, rtol=1e-11)
    actual_check, obs_actual, difference_actual, _ = observable_forward(actual_raw)
    np.testing.assert_allclose(actual_check, actual, atol=1e-11, rtol=1e-11)
    center, scale = np.asarray(stats['u_center']), np.asarray(stats['u_scale'])
    u, target = (raw-center)/scale, (actual_raw-center)/scale
    cc, cw, cl, cu = interval_levels(u, target)
    gc, gw, gl, gu = interval_levels(gamma, actual)
    oc, ow, _, _ = interval_levels(obs, obs_actual)
    crps, cover, width = interval_scores(obs, obs_actual)
    absolute = np.log1p(difference*norm2[None, :, None])
    absolute_actual = np.log1p(difference_actual*norm2[:, None])
    ac, av, _ = interval_scores(absolute, absolute_actual)
    p_null = (gamma <= 0).mean(0)
    out = dict(predicted=gamma.mean(0), p_null=p_null,
        crps=fair_crps(gamma, actual), brier=(p_null-(actual <= 0))**2,
        mean_u=u.mean(0), actual_u=target,
        mean_u_mc_se=u.std(0, ddof=1)/np.sqrt(samples),
        energy=np.linalg.norm(u-target[None], axis=-1).mean(0)
            -.5*np.linalg.norm(u[:samples//2]-u[samples//2:], axis=-1).mean(0),
        coordinate_coverage_by_level=cc.mean(1), coordinate_width_by_level=cw.mean(1),
        coordinate_lower=cl[3], coordinate_upper=cu[3],
        coordinate_coverage=cc[..., 3].mean(1),
        gamma_coverage_by_level=gc, gamma_width_by_level=gw,
        coverage=gc[..., 3], gamma_lower=gl[3], gamma_upper=gu[3],
        gamma_mc_se=gamma.std(0, ddof=1)/np.sqrt(samples),
        null_mc_se=np.sqrt(p_null*(1-p_null)/samples),
        observable_crps=crps, observable_coverage=cover, observable_width=width,
        observable_coverage_by_level=oc, observable_width_by_level=ow,
        absolute_crps_by_pair=ac, absolute_coverage_by_pair=av,
        absolute_pair_crps=ac.mean(1), absolute_pair_coverage=av.mean(1),
        predicted_cos_Z1_V=cosine[..., 0].mean(0), predicted_cos_Z2_V=cosine[..., 1].mean(0))
    for prefix, cols in (('single', slice(0, 3)), ('pair', slice(3, 6)),
                         ('average', slice(6, 10)), ('triple_average', slice(9, 10))):
        out[prefix+'_crps'] = crps[:, cols].mean(1)
        out[prefix+'_coverage'] = cover[:, cols].mean(1)
    if any(not np.isfinite(value).all() or len(value) != n for value in out.values()):
        raise ValueError('Invalid per-object score')
    return out, gamma


def paired_summary(left, right, data, actual):
    values = {'gamma_mse': ((left['predicted']-actual)**2, (right['predicted']-actual)**2),
              'brier': ((left['p_null']-(actual <= 0))**2, (right['p_null']-(actual <= 0))**2),
              'geometry_mse': (np.square(left['mean_u']-left['actual_u']).mean(1),
                               np.square(right['mean_u']-right['actual_u']).mean(1))}
    for key in ('crps', 'energy', 'single_crps', 'pair_crps', 'average_crps', 'triple_average_crps', 'absolute_pair_crps'):
        values[key] = (left[key], right[key])
    for lam in (.2, 0.):
        key = f'lambda_{lam:g}'
        a, b = left['selected_'+key], right['selected_'+key]
        values['policy_value_per_candidate_'+key] = (actual*a, actual*b)
        values['false_activation_per_candidate_'+key] = (
            comparison.false_activation_outcomes(actual, a), comparison.false_activation_outcomes(actual, b))
    return {key: {scope: bootstrap_difference(a, b, data[scope])
                  for scope in ('groups', 'layout')} for key, (a, b) in values.items()}


def calibration_and_overlap(left, right, data, actual, folds):
    result = {}
    for lam in (.2, 0.):
        key = f'lambda_{lam:g}'
        ls, rs = left['selected_'+key], right['selected_'+key]
        cells = {}
        for arm, out, own, other in ((ARM, left, ls, rs), ('CORE_ORIGINAL', right, rs, ls)):
            regions = []
            for scope, take in (('all', np.ones(len(actual), bool)), ('own_selected', own), ('other_selected', other)):
                gap = ((actual <= 0)-out['p_null'])*take
                regions.append(dict(scope=scope, n=int(take.sum()),
                    predicted_null=float(out['p_null'][take].sum()), actual_null=int((actual[take] <= 0).sum()),
                    brier=float(np.mean((out['p_null'][take]-(actual[take] <= 0))**2)),
                    gap_per_candidate={label: bootstrap_difference(gap, np.zeros(len(gap)), data[label])
                                       for label in ('groups', 'layout')}))
            rankbands = []
            for lower, upper in ((0, .125), (.125, .25), (.25, .5), (.5, 1.)):
                take = np.zeros(len(actual), bool)
                for f in np.unique(folds):
                    q = np.flatnonzero(folds == f)
                    ordered = q[np.lexsort((data['ids'][q], -(out['predicted'][q]-lam*out['p_null'][q])))]
                    take[ordered[int(lower*len(q)):int(upper*len(q))]] = True
                rankbands.append(dict(lower=lower, upper=upper, n=int(take.sum()),
                    mean_p=float(out['p_null'][take].mean()), actual_rate=float((actual[take] <= 0).mean())))
            cells[arm] = dict(regions=regions, score_rank_bands=rankbands)
        result[key] = dict(calibration=cells, overlap=dict(shared=int((ls&rs).sum()),
            L_n=int(ls.sum()), CORE_n=int(rs.sum()), jaccard=float((ls&rs).sum()/(ls|rs).sum()),
            list_symmetric_difference=int((ls != rs).sum())))
    return result


def analyze(data, core, observed, parts, seed_offset=0):
    blocks, seconds = [], []
    for f, part in enumerate(parts):
        q = part['DEV_EVAL']
        for start in range(0, len(q), OBJECT_CHUNK):
            path = ROOT/f'fold_{f}'/f'seed_{seed_offset}'/f'block_{start:04}.npz'
            saved = comparison.read_npz(path)
            np.testing.assert_array_equal(saved['ids'], data['ids'][q[start:start+OBJECT_CHUNK]])
            blocks.append((q[start:start+OBJECT_CHUNK], saved))
            seconds.append(json.loads(path.with_suffix('.json').read_text())['wall_seconds'])
    keys = set(blocks[0][1])-{'ids', 'actual', 'gamma_samples'}
    out = {key: np.empty((len(data['ids']), *blocks[0][1][key].shape[1:]), dtype=blocks[0][1][key].dtype) for key in keys}
    for q, saved in blocks:
        np.testing.assert_array_equal(saved['actual'], observed['actual'][q])
        for key in keys:
            out[key][q] = saved[key]
    actual, folds = observed['actual'], core['fold']
    metrics = {ARM: comparison.summarize(out, actual, data['ids'], folds),
               'CORE_ORIGINAL': comparison.summarize(core, actual, data['ids'], folds)}
    metrics[ARM]['profile_joint_nll_separate_space'] = float(out['profile_joint_nll'].mean())
    for key in ('gamma_mc_se', 'null_mc_se', 'coordinate_coverage_by_level', 'coordinate_width_by_level', 'gamma_width_by_level', 'observable_crps', 'observable_coverage_by_level'):
        metrics[ARM][key] = out[key].mean(0)
    resources = []
    for f, part in enumerate(parts):
        costs = resource_counts(part)
        values = {}
        for arm, scored in ((ARM, out), ('CORE_ORIGINAL', core)):
            selected = scored['selected_lambda_0.2'] & (folds == f)
            net = float(actual[selected].sum())
            refcost = .01*costs['CORE_reference_new_wells'] if arm == 'CORE_ORIGINAL' else 0.
            fitcost = costs['CORE_fully_new_fit_well_cost'] if arm == 'CORE_ORIGINAL' else costs['L_fully_new_fit_well_cost']
            values[arm] = dict(action_net_value=net, existing_resources_net=net,
                new_reference_only_net=net-refcost, fully_new_fit_resources_net=net-fitcost)
        resources.append(dict(fold=f, counts=costs, value_scenarios=values))
    payload = dict(complete=True, seed_offset=seed_offset, samples=SAMPLES, n=len(actual),
        metrics=metrics, paired_L_minus_CORE=paired_summary(out, core, data, actual),
        policies=calibration_and_overlap(out, core, data, actual, folds), resources=resources,
        MC_and_scoring_wall_seconds=sum(seconds),
        common_nll_available=False, common_joint_coverage_available=False,
        historical_L_used_for_comparison=False, existing_31_arms_refitted=False,
        purpose='primary comparison' if seed_offset == 0 else 'post-fit Monte Carlo sensitivity',
        additional_model_fits=0 if seed_offset else len(parts),
        additional_label_wells=0 if seed_offset else sum(4*len(part['TRAIN']) for part in parts),
        resources_are_incremental=seed_offset == 0,
        resource_note='Original fit-resource scenarios; sensitivity seeds add computation only, not training or deployment label cost',
        scope='Fixed-model paired development diagnostics; two layouts only, not population confidence or certification',
        final_opened=False, fifth_repeat_opened=False)
    suffix = '' if seed_offset == 0 else f'_mc{seed_offset}'
    save_npz(ROOT/(ARM+suffix+'.npz'), ids=data['ids'], groups=data['groups'], layout=data['layout'], fold=folds, actual=actual, **out)
    write_json(ROOT/('summary'+suffix+'.json'), payload)
    write_json(REPORT/('summary'+suffix+'.json'), payload)
    lines = ['# JUMP matched complete L', '',
        'Original complete L refitted on the exact 246 TRAIN identities in each existing fold. Full 3617-dimensional residual sampling, 100,000 draws per query. Existing 31 arms were read from cache.', '',
        '| Arm | Gamma CRPS | NULL Brier | NULL AUC | Gamma rho | Selected NULL / 79 | Selected mean Gamma |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, metric in metrics.items():
        policy = metric['policies']['lambda_0.2']
        lines.append(f"| {name} | {metric['crps']:.6f} | {metric['brier']:.6f} | {metric['null_auc']:.4f} | {metric['gamma_spearman']:.4f} | {policy['null_selected']} | {policy['selected_mean_value']:.6f} |")
    lines += ['', 'The same label-resource ceiling is available to both methods. L uses 984 TRAIN wells per fold; it leaves the 61 VALID, 102/103 REF and 102 CAL objects unused. CORE uses or selects on those additional resources. This is availability matching, not equal actual label consumption.',
        'The L conditional mean depends on its covariance. Updating its moments on REF while calling the conditional mean TRAIN-only would change the model and was not done.',
        'All induced-geometry sample scores use CORE\'s saved TRAIN-standardized coordinates. Full-profile Gaussian NLL is reported separately and never compared with the 9-dimensional CORE NLL. No analytic geometry NLL or joint-region coverage is invented.',
        'Lambda 0.2 and 0 selection lists, selected-region calibration, paired chemical-identity and layout resampling, resource scenarios and computation times are in summary.json. Layout resampling over two plate tuples is descriptive.',
        'FINAL and the fifth repeat remain closed.']
    (ROOT/('REPORT'+suffix+'.md')).write_text('\n'.join(lines)+'\n')
    (REPORT/('REPORT'+suffix+'.md')).write_text('\n'.join(lines)+'\n')
    return payload


def run(seed_offset=0, prepare_only=False, reuse_fitted_only=False):
    started = time.monotonic()
    reuse_fitted_only = reuse_fitted_only or seed_offset != 0
    if reuse_fitted_only:
        missing = [str(ROOT/f'fold_{f}'/'complete_original_L.npz') for f in range(5)
                   if not (ROOT/f'fold_{f}'/'complete_original_L.npz').is_file()]
        if missing:
            raise ValueError('Post-fit integration requires all existing L fits; no refit is allowed: '+', '.join(missing))
    data, core, observed, parts, source_manifest = load_inputs()
    plan = freeze_plan(parts, source_manifest)
    if prepare_only:
        print(json.dumps(dict(ready=True, resource_counts=plan['resources']), indent=2))
        return
    status_path = ROOT/f'status_seed_{seed_offset}.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state') == 'COMPLETE':
        print(status_path.read_text())
        return
    def status(state, **extra):
        record = dict(state=state, pid=os.getpid(), seed_offset=seed_offset,
            elapsed_this_invocation_seconds=time.monotonic()-started, **extra)
        write_json(status_path, record)
        if seed_offset == 0:
            write_json(ROOT/'status.json', record)
        print(json.dumps(record), flush=True)
    completed = 0
    try:
        for f, part in enumerate(parts):
            folder = ROOT/f'fold_{f}'
            folder.mkdir(exist_ok=True)
            fitted = folder/'complete_original_L.npz'
            if fitted.exists():
                model = ClosedFormBaseline.load(fitted)
                if model.metadata['train_ids'] != data['ids'][part['TRAIN']].tolist():
                    raise ValueError('Saved L training identities changed')
            else:
                if reuse_fitted_only:
                    raise ValueError('An existing L fit disappeared; post-fit integration never retrains')
                before = time.monotonic()
                model = fit_original_l(data['Y'][part['TRAIN']], data['ids'][part['TRAIN']], SEED+100*f)
                model.save(fitted)
                write_json(folder/'fit_summary.json', dict(wall_seconds=time.monotonic()-before,
                    cpu_threads=1, diagnostics=model.diagnostics, resources=resource_counts(part)))
            stats = json.loads((SOURCE/f'fold_{f}/mean/preprocessing.json').read_text())
            q = part['DEV_EVAL']
            destination = folder/f'seed_{seed_offset}'
            destination.mkdir(exist_ok=True)
            for start in range(0, len(q), OBJECT_CHUNK):
                rows = q[start:start+OBJECT_CHUNK]
                path = destination/f'block_{start:04}.npz'
                if path.exists() and path.with_suffix('.json').exists():
                    completed += len(rows)
                    continue
                block_seed = SEED+f*100+seed_offset+1000000*start
                status('RUNNING', stage='full_profile_sampling', fold=f, block_start=start, completed_objects=completed, total_objects=len(data['ids']))
                before = time.monotonic()
                grams, mean_gram = sample_original_l(model, data['Y'][rows, 0], samples=SAMPLES, seed=block_seed)
                sampled_seconds = time.monotonic()-before
                out, gamma = score_draws(grams, observed['raw_geometry'][rows], observed['actual'][rows], stats,
                    np.square(data['Y'][rows, 0]).mean(1))
                out['mean_profile_gram'] = mean_gram
                out['profile_joint_nll'] = -model.conditional(data['Y'][rows, :1], [0], [1, 2, 3]).log_prob(data['Y'][rows, 1:])
                save_npz(path, ids=data['ids'][rows], actual=observed['actual'][rows], gamma_samples=gamma, **out)
                write_json(path.with_suffix('.json'), dict(seed=block_seed, samples=SAMPLES,
                    query_ids=data['ids'][rows], wall_seconds=time.monotonic()-before,
                    full_profile_sampling_seconds=sampled_seconds, cpu_threads=1,
                    draw_chunk_size=DRAW_CHUNK, object_chunk_size=OBJECT_CHUNK))
                completed += len(rows)
            status('RUNNING', stage='fold_complete', fold=f, completed_objects=completed, total_objects=len(data['ids']))
        status('RUNNING', stage='paired_analysis', completed_objects=completed)
        result = analyze(data, core, observed, parts, seed_offset)
        suffix = '' if seed_offset == 0 else f'_mc{seed_offset}'
        status('COMPLETE', completed_objects=completed, report=str(ROOT/('REPORT'+suffix+'.md')),
               total_MC_and_scoring_wall_seconds=result['MC_and_scoring_wall_seconds'])
    except BaseException:
        status('FAILED', completed_objects=completed, traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--seed-offset', type=int, choices=(0, 100000, 200000), default=0)
    parser.add_argument('--reuse-fitted-only', action='store_true', help='Require all five existing L fits; never train')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args.seed_offset, args.prepare_only, args.reuse_fitted_only)
