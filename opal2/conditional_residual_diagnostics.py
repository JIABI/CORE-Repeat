"""Post-evaluation angular and centering diagnostics for frozen CORE residuals.

This companion reads completed predictions only. It fits no models, selects no
policy and never overwrites the evaluation runner's files. Projection labels
refer to geometry, not physically identified independent/shared noise.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .biology_borrowing_experiment import read_json, read_npz
from .biology_kernel_evaluation import plain
from .joint_contrast_scale import contrast_projector
from .lincs_biology_experiment import load_data


def projected_error_diagnostics(mean, actual, raw_mean, target_scale, scatter, covariances):
    """Observed error and forecast second moments in one common CORE frame.

    ``expected_energy_fraction`` is a ratio of expected quadratic energies. It
    is NOT the expectation of the random angular fraction E3/(E3+E6).
    """
    mean, actual = np.asarray(mean, float), np.asarray(actual, float)
    if mean.ndim != 2 or mean.shape[1] != 9 or actual.shape != mean.shape:
        raise ValueError('Aligned nine-coordinate mean and actual values required')
    if not np.isfinite(mean).all() or not np.isfinite(actual).all():
        raise ValueError('Finite fixed means and observed outcomes required')
    decomposition = contrast_projector(raw_mean, target_scale, scatter)
    L, P = decomposition['factor'], decomposition['projector']
    w = np.linalg.solve(L, (actual-mean)[..., None])[..., 0]
    wp = np.einsum('nij,nj->ni', P, w)
    energy = np.column_stack((np.square(wp).sum(1), np.square(w-wp).sum(1)))
    total = energy.sum(1)
    fraction = np.divide(energy[:, 0], total, out=np.full(len(w), np.nan), where=total > 0)
    expected = {}
    for name, covariance in covariances.items():
        covariance = np.asarray(covariance, float)
        if covariance.shape != (len(w), 9, 9) or not np.isfinite(covariance).all():
            raise ValueError('Aligned finite arm covariance_u required: '+name)
        if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-11, atol=1e-12):
            raise ValueError('Arm covariance is not symmetric: '+name)
        np.linalg.cholesky(covariance)
        left = np.linalg.solve(L, covariance)
        white = np.linalg.solve(L, left.swapaxes(-1, -2)).swapaxes(-1, -2)
        first = np.einsum('nij,nji->n', P, white)
        remainder = np.trace(white, axis1=1, axis2=2)-first
        if np.any(first <= 0) or np.any(remainder <= 0):
            raise ValueError('Nonpositive expected projection energy: '+name)
        expected[name] = dict(energies=np.column_stack((first, remainder)),
            expected_energy_fraction=first/(first+remainder))
    return dict(whitened_residual=w, energies=energy, radius=np.sqrt(total),
                angular_fraction=fraction, expected=expected)


def amplitude_strata(training_log_amplitude, query_log_amplitude):
    train, query = np.asarray(training_log_amplitude, float), np.asarray(query_log_amplitude, float)
    if train.ndim != 1 or query.ndim != 1 or len(train) < 4 or not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError('Finite training-only amplitude and query amplitude vectors required')
    edges = np.quantile(train, [.25, .5, .75])
    labels = np.searchsorted(edges, query, side='right')
    return labels, dict(quantiles=[.25, .5, .75], edges=edges.tolist(),
        training_min=float(train.min()), training_max=float(train.max()),
        training_n=len(train), fitted_from='original MODEL_FIT X only',
        tie_rule='equal-to-edge enters upper stratum; duplicate edges may leave empty strata')


def _moment_summary(w):
    """Finite-cohort centering check; a nonzero mean is not identified biology."""
    w = np.asarray(w, float)
    if not len(w):
        return None
    center = w.mean(0)
    total = np.square(w).sum(1).mean()
    center_energy = float(center@center)
    centered = np.square(w-center).sum(1).mean()
    return dict(whitened_residual_mean=center.tolist(), whitened_mean_norm=float(np.linalg.norm(center)),
        mean_vector_squared_norm=center_energy, mean_uncentered_total_energy=float(total),
        mean_centered_total_energy=float(centered),
        finite_cohort_mean_energy_fraction=float(center_energy/total) if total > 0 else None,
        decomposition_roundoff=float(total-center_energy-centered),
        interpretation='Observed mean prediction error, not an identified physical-noise component or a bias significance test')


def summarize_subset(arrays, mask):
    mask = np.asarray(mask, bool)
    n = int(mask.sum())
    if not n:
        return dict(n=0)
    energy, fraction = arrays['energies'][mask], arrays['angular_fraction'][mask]
    finite = np.isfinite(fraction)
    observed_total = float(energy.sum())
    out = dict(n=n, observed_mean_energies=energy.mean(0).tolist(),
        observed_mean_energy_per_degree=(energy.mean(0)/[3, 6]).tolist(),
        observed_ratio_of_pooled_energies=float(energy[:, 0].sum()/observed_total) if observed_total > 0 else None,
        observed_mean_angular_fraction=float(fraction[finite].mean()) if finite.any() else None,
        observed_zero_error_rows=int((~finite).sum()),
        centering=_moment_summary(arrays['whitened_residual'][mask]), models={})
    for arm, values in arrays['expected'].items():
        expected = values['energies'][mask]
        out['models'][arm] = dict(expected_mean_energies=expected.mean(0).tolist(),
            expected_mean_energy_per_degree=(expected.mean(0)/[3, 6]).tolist(),
            expected_ratio_of_pooled_energies=float(expected[:, 0].sum()/expected.sum()),
            mean_ratio_of_expected_energies=float(values['expected_energy_fraction'][mask].mean()),
            observed_minus_expected_mean_energies=(energy-expected).mean(0).tolist(),
            per_object_energy_squared_error=np.square(energy-expected).mean(0).tolist())
    return out


def _finite_list(values):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values).reshape(-1)]


def build_diagnostics(run):
    root = Path(run).resolve()
    status = read_json(root/'status.json')
    summary = read_json(root/'summary.json')
    if status.get('state') != 'COMPLETE' or summary.get('state') != 'COMPLETE':
        raise ValueError('Wait for the complete evaluation; partial arms are not pooled')
    spec = read_json(root/'run_spec.json')
    source = Path(spec['source'])
    radial = Path(read_json(source/'summary.json')['source'])
    radial_summary = read_json(radial/'summary.json')
    original = Path(radial_summary['reference_run'])
    manifest = read_json(original/'run_manifest.json')
    data, metadata = load_data(radial_summary['data_directory'])
    ids = data['ids']; n = len(ids)
    prior = read_npz(radial/'AMP_EMP_LOCAL.npz')
    np.testing.assert_array_equal(prior['ids'], ids)
    arms = {a: read_npz(root/(a+'.npz')) for a in spec['arms']}
    for name, values in arms.items():
        np.testing.assert_array_equal(values['ids'], ids)
        np.testing.assert_array_equal(values['mean_u'], prior['mean_u'])
        np.testing.assert_array_equal(values['actual_u'], prior['actual_u'])
    amplitude = np.log(np.sqrt(np.square(data['Y'][:, 0]).mean(1)))
    if not np.isfinite(amplitude).all():
        raise ValueError('First-well amplitude is nonfinite')
    arrays = dict(whitened_residual=np.empty((n, 9)), energies=np.empty((n, 2)),
        radius=np.empty(n), angular_fraction=np.empty(n),
        expected={a: dict(energies=np.empty((n, 2)), expected_energy_fraction=np.empty(n)) for a in arms})
    amplitude_bin, count = np.full(n, -1, int), np.zeros(n, int)
    boundaries = []; folds = arms['CORE']['fold'].astype(int)
    for record in manifest['folds']:
        fold = int(record['fold']); query = np.flatnonzero(folds == fold)
        fit = np.asarray(record['fit'], int)
        if set(data['groups'][fit]) & set(data['groups'][query]):
            raise ValueError('Amplitude boundary training contains query groups')
        stats = read_json(original/'folds'/f'fold_{fold}'/'preprocessing.json')
        scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
        result = projected_error_diagnostics(prior['mean_u'][query], prior['actual_u'][query],
            prior['mean_u'][query]*scale+center, scale, prior['scatter_u'][query],
            {a: value['covariance_u'][query] for a, value in arms.items()})
        for key in ('whitened_residual', 'energies', 'radius', 'angular_fraction'):
            arrays[key][query] = result[key]
        for a in arms:
            for key in ('energies', 'expected_energy_fraction'):
                arrays['expected'][a][key][query] = result['expected'][a][key]
        amplitude_bin[query], edge_record = amplitude_strata(amplitude[fit], amplitude[query])
        boundaries.append(dict(fold=fold, **edge_record)); count[query] += 1
    np.testing.assert_array_equal(count, np.ones(n, int))
    # Radius strata explicitly use evaluated outcomes and cannot be rule inputs.
    quantiles = [.5, .8, .9, .95, .99]
    radius_edges = np.quantile(arrays['radius'], quantiles)
    radius_bin = np.searchsorted(radius_edges, arrays['radius'], side='right')
    full = summarize_subset(arrays, np.ones(n, bool))
    by_amplitude = [dict(stratum=k, **summarize_subset(arrays, amplitude_bin == k)) for k in range(4)]
    by_radius = [dict(stratum=k, **summarize_subset(arrays, radius_bin == k)) for k in range(6)]
    by_fold = [dict(fold=f, **summarize_subset(arrays, folds == f)) for f in sorted(set(folds))]
    objects = []
    for i, name in enumerate(ids):
        objects.append(dict(id=str(name), group=str(data['groups'][i]), fold=int(folds[i]),
            layout=str(metadata['units'][i]['layout_block']), log_rms=float(amplitude[i]),
            amplitude_stratum=int(amplitude_bin[i]), outcome_radius_stratum=int(radius_bin[i]),
            core_whitened_residual=arrays['whitened_residual'][i].tolist(),
            observed_projection_energies=arrays['energies'][i].tolist(),
            observed_core_radius=float(arrays['radius'][i]),
            observed_angular_fraction=_finite_list([arrays['angular_fraction'][i]])[0],
            models={a: dict(expected_projection_energies=arrays['expected'][a]['energies'][i].tolist(),
                ratio_of_expected_energies=float(arrays['expected'][a]['expected_energy_fraction'][i])) for a in arms}))
    return dict(status='COMPLETE', created_utc=datetime.now(timezone.utc).isoformat(),
        run=str(root), prior_radial_source=str(radial), n=n, arms=list(arms), full=full,
        amplitude_edges_by_fold=boundaries, by_amplitude=by_amplitude,
        outcome_radius_quantiles=dict(probabilities=quantiles, edges=radius_edges.tolist(),
            use='Descriptive outcome-based diagnostics only; never inputs, selection or fitting'),
        by_outcome_radius=by_radius, by_fold=by_fold, objects=objects,
        labels=['pair-difference-sensitive geometric directions (rank 3)', 'remaining geometric directions (rank 6)'],
        notes=[
            'CORE scatter, not each arm covariance, defines the common whitening and projection frame.',
            'Arm covariance_u includes the saved empirical radial second-moment multiplier.',
            'Ratio of expected energies is not expected angular fraction; these are not interchanged.',
            'Outcome-radius stratification induces selection by observed error; bin-wise forecast moment gaps are descriptive, not calibration tests.',
            'Fixed prediction means can retain conditional bias. Residual energy combines bias and dispersion, not pure stochastic noise.',
            'NO_CONTROL_CONTEXT removes plate_controls and position_qc only; dose/time/cell-background context remains.',
            'This moment diagnostic cannot establish a fully correct within-object radius-direction joint distribution.',
            'No model, checkpoint, descriptor, threshold or policy was selected using these outcomes.'])


def write_diagnostics(run, output=None):
    root = Path(run).resolve()
    out = Path(output).resolve() if output else root/'conditional_residual_diagnostics'
    if out.exists():
        raise FileExistsError('Preserve existing diagnostics; supply a fresh output directory')
    result = build_diagnostics(root)
    out.mkdir(parents=True)
    (out/'diagnostics.json').write_text(json.dumps(plain(result), indent=2, allow_nan=False)+'\n')
    full = result['full']
    lines = ['# Conditional residuals: angular moments and residual centering', '',
        f"Completed read-only diagnostic for {result['n']} opened LINCS objects. CORE means are fixed.", '',
        'The two projection blocks are geometric, not physical independent/shared noise components.', '',
        '| Arm | Expected rank-3 energy | Expected rank-6 energy | Ratio of pooled expected energies |',
        '|---|---:|---:|---:|']
    lines.append(f"| Observed | {full['observed_mean_energies'][0]:.6f} | {full['observed_mean_energies'][1]:.6f} | {full['observed_ratio_of_pooled_energies']:.6f} |")
    for arm, m in full['models'].items():
        e = m['expected_mean_energies']
        lines.append(f"| {arm} | {e[0]:.6f} | {e[1]:.6f} | {m['expected_ratio_of_pooled_energies']:.6f} |")
    c = full['centering']
    lines += ['', '## Residual centering', '',
        f"Mean CORE-whitened error vector: {c['whitened_residual_mean']}.",
        f"Its norm is {c['whitened_mean_norm']:.6f}; finite-cohort mean-vector energy is {c['mean_vector_squared_norm']:.6f}.",
        f"Uncentered mean energy {c['mean_uncentered_total_energy']:.6f} = centered mean energy {c['mean_centered_total_energy']:.6f} + mean-vector energy.",
        'This is a descriptive mean-error check, not proof of a biological/technical noise component or a test that bias is nonzero.', '',
        '## Error-radius strata (outcome-based, descriptive only)', '',
        '| Quantile stratum | n | Observed mean angular fraction | Observed pooled energy fraction | CORE forecast energy fraction | DESCRIPTORS | DESCRIPTORS_LATENT |',
        '|---|---:|---:|---:|---:|---:|---:|']
    labels = ('0–50%', '50–80%', '80–90%', '90–95%', '95–99%', '99–100%')
    for label, item in zip(labels, result['by_outcome_radius']):
        if not item['n']:
            continue
        values = [item['models'][a]['expected_ratio_of_pooled_energies'] for a in ('CORE','DESCRIPTORS','DESCRIPTORS_LATENT')]
        lines.append(f"| {label} | {item['n']} | {item['observed_mean_angular_fraction']:.6f} | {item['observed_ratio_of_pooled_energies']:.6f} | "+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', 'Observed angular fractions and ratios of expected energies are different quantities. '
        'Selecting bins by realized radius changes the observed error distribution; these are not bin-wise calibration guarantees.', '',
        'Amplitude-quartile edges were fitted to the original MODEL_FIT X values, separately by fold. '
        'Full amplitude/fold summaries and matched per-object values are in diagnostics.json.', '',
        'NO_CONTROL_CONTEXT removes normalized plate-control and position/QC blocks only. '
        'It retains dose, time and cell-background covariates.', '', *['- '+v for v in result['notes']]]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output')
    parser.add_argument('--wait-complete', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args(); root = Path(args.run).resolve()
    if args.launch:
        record = root/'conditional_diagnostic_process.json'
        if record.exists():
            raise FileExistsError('Diagnostic waiter already launched')
        command = [sys.executable, '-u', '-m', __spec__.name, '--run', str(root), '--wait-complete']
        if args.output:
            command += ['--output', args.output]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (root/'conditional_diagnostic_stdout.log').open('xb') as out, (root/'conditional_diagnostic_stderr.log').open('xb') as err:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], env=env,
                                       stdout=out, stderr=err, start_new_session=True)
        record.write_text(json.dumps(dict(pid=process.pid, command=command), indent=2)+'\n')
        print('Diagnostic waiter launched', process.pid, flush=True)
        return
    if args.wait_complete:
        while True:
            if (root/'status.json').exists():
                state = read_json(root/'status.json').get('state')
                if state == 'FAILED':
                    raise RuntimeError('Evaluation failed; diagnostic did not read partial outcomes')
                if state == 'COMPLETE' and (root/'summary.json').exists():
                    break
            time.sleep(60)
    print('Completed diagnostic:', write_diagnostics(root, args.output), flush=True)


if __name__ == '__main__':
    main()
