"""Append TRAIN-scaled amplitude controls to the saved covariance diagnostic.

Original results are read-only. Exact original node bootstrap draws are reused.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
from threadpoolctl import threadpool_limits

from opal2.relation_variance_components import (
    KERNEL_NAMES, group_dyad_pairs, node_pair_weights,
    solve_moments, partial_design_fraction,
)
from scripts.diagnose_r3_relation_variance_20260921 import (
    OUTPUT, OLD_RUN, load_units, read_npz, write_json, interval,
)

NAMES = KERNEL_NAMES + ('same_amplitude_bin', 'continuous_amplitude')
MODELS = {
    'ADJUSTED_BINS': (0, 1, 3, 4, 5),
    'ADJUSTED_CONTINUOUS': (0, 1, 3, 4, 6),
    'ADJUSTED_BOTH': (0, 1, 3, 4, 5, 6),
    'SIGNED_JOINT_BINS': (0, 1, 2, 3, 4, 5),
    'SIGNED_JOINT_CONTINUOUS': (0, 1, 2, 3, 4, 6),
    'SIGNED_JOINT_BOTH': (0, 1, 2, 3, 4, 5, 6),
}


def amplitude_features(log_norm, edges):
    """TRAIN q20/q40/q60/q80 define all feature scaling, never query moments."""
    edges = np.asarray(edges, float)
    if edges.shape != (4,) or not np.isfinite(edges).all() or np.any(np.diff(edges) < 0):
        raise ValueError('Four ordered saved TRAIN quintile edges required')
    values = np.asarray(log_norm, float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Finite source log norms required')
    center = float((edges[1] + edges[2]) / 2.)
    scale = float(edges[3] - edges[0])
    if scale <= 1e-12:
        raise ValueError('TRAIN amplitude q80-q20 cannot scale the continuous kernel')
    bins = np.searchsorted(edges, values, side='right')
    z = (values-center) / scale
    return bins, z, center, scale


def add_amplitude_design(units):
    """Recover identical annotation/zero-X selection through saved metadata."""
    from opal2.rxrx3_r3_biology import load_rxrx3_biology_metadata
    keys = ('ids', 'groups', 'object_ids', 'dose', 'batches', 'well_ids', 'plates', 'layout')
    data = read_npz(PROJECT/'data/rxrx3_r2_20260918/prepared_r2/data.npz', keys)
    biology = load_rxrx3_biology_metadata(data)
    lookup = {str(cid): i for i, cid in enumerate(data['ids'])}
    audits = []
    for number, unit in enumerate(units):
        folder = OLD_RUN/f'unit_{number:02d}'
        q = read_npz(folder/'query_scores.npz', ('source_ids', 'groups'))
        source = np.array([lookup[str(cid)] for cid in q['source_ids']])
        x = read_npz(folder/'predictions.npz', ('source_x',))['source_x'].astype(float)
        norm = np.linalg.norm(x, axis=1)
        take = biology['arrays']['target_mask'][source] & (norm > 1e-12)
        groups = q['groups'][take]
        i, j, w = group_dyad_pairs(groups)
        np.testing.assert_array_equal(groups[i], unit['left'])
        np.testing.assert_array_equal(groups[j], unit['right'])
        np.testing.assert_array_equal(w, unit['w'])
        edges = read_npz(folder/'references.npz', ('amplitude_bin_edges',))['amplitude_bin_edges']
        log_norm = np.log(norm[take])
        bins, z, center, scale = amplitude_features(log_norm, edges)
        unit['amp'] = np.column_stack(((bins[i] == bins[j]).astype(float), z[i]*z[j]))
        _, inverse, count = np.unique(groups, return_inverse=True, return_counts=True)
        row_weight = 1. / (len(count)*count[inverse])
        audits.append(dict(unit=number, edges=edges, center=center, scale=scale,
            center_rule='(TRAIN q40+q60)/2; a quantile-based center, not the exact TRAIN median',
            scale_rule='TRAIN q80-q20', n_rows=len(groups), n_groups=len(count),
            bin_counts=np.bincount(bins, minlength=5),
            continuous_feature_min=float(z.min()), continuous_feature_max=float(z.max()),
            continuous_kernel_group_equal_mean_diagonal=float(row_weight @ np.square(z)),
            bin_kernel_mean_diagonal=1.,
            amplitude_features_fitted_to_query=False))
    return audits


def augmented_moments(units, original_g, original_h, multiplicity, groups, chunk=32):
    n, n_units = original_g.shape[:2]
    g = np.zeros((n, n_units, 7, 7))
    h = np.zeros((n, n_units, 7, 2))
    g[:, :, :5, :5], h[:, :, :5] = original_g, original_h
    index = {str(group): i for i, group in enumerate(groups)}
    for u, unit in enumerate(units):
        left = np.array([index[str(a)] for a in unit['left']])
        right = np.array([index[str(a)] for a in unit['right']])
        x, a, y = unit['x'], unit['amp'], unit['y']
        xa = (x[:, :, None] * a[:, None, :]).reshape(len(x), -1)
        aa = (a[:, :, None] * a[:, None, :]).reshape(len(x), -1)
        ay = (a[:, :, None] * y[:, None, :]).reshape(len(x), -1)
        for start in range(0, n, chunk):
            stop = min(start+chunk, n)
            w = node_pair_weights(multiplicity[start:stop], left, right, unit['w'])
            w /= w.sum(axis=1)[:, None]
            mx, ma, my = w@x, w@a, w@y
            cross = (w@xa).reshape(-1, 5, 2)-mx[:, :, None]*ma[:, None, :]
            g[start:stop, u, :5, 5:] = cross
            g[start:stop, u, 5:, :5] = cross.transpose(0, 2, 1)
            g[start:stop, u, 5:, 5:] = (w@aa).reshape(-1, 2, 2)-ma[:, :, None]*ma[:, None, :]
            h[start:stop, u, 5:] = (w@ay).reshape(-1, 2, 2)-ma[:, :, None]*my[:, None, :]
    np.testing.assert_array_equal(g[:, :, :5, :5], original_g)
    np.testing.assert_array_equal(h[:, :, :5], original_h)
    return g, h


def summarize(original, g, h, e):
    result, flat, saved = {}, [], {}
    scope_units = {'ALL_SEVEN_PAIRS': list(range(35))}
    scope_units.update({f'PAIR_{p}': [u for u in range(35) if u % 7 == p] for p in range(7)})
    for scope, selected in scope_units.items():
        gg, hh, energy = g[:, selected].mean(1), h[:, selected].mean(1), e[:, selected].mean(1)
        result[scope] = {}
        for task, task_name in enumerate(('SAME', 'CROSS')):
            result[scope][task_name] = {}
            for model, indices in MODELS.items():
                fits = [solve_moments(gg[b], hh[b, :, task], indices) for b in range(len(gg))]
                raw = np.array([f['raw'] for f in fits])
                bounded = np.array([f['nonnegative'] for f in fits])
                baseline_name = 'SIGNED_JOINT' if model.startswith('SIGNED') else 'ADJUSTED'
                baseline = original['scopes'][scope]['outcomes'][task_name]['models'][baseline_name]
                record = dict(rank=fits[0]['rank'], n_components=len(indices),
                    identifiable=fits[0]['identifiable'], condition_number=fits[0]['condition_number'],
                    kernel_names=[NAMES[k] for k in indices],
                    kernel_correlation=fits[0]['kernel_correlation'],
                    bootstrap_rank_deficient_count=sum(not f['identifiable'] for f in fits[1:]),
                    target_residual_design_fraction=partial_design_fraction(gg[0], 0, tuple(i for i in indices if i != 0)),
                    signed_residual_design_fraction=partial_design_fraction(gg[0], 2, tuple(i for i in indices if i != 2)) if 2 in indices else None,
                    components={})
                for j, idx in enumerate(indices):
                    name = NAMES[idx]
                    val = dict(raw=float(raw[0, j]), raw_ci95=interval(raw[1:, j]),
                        nonnegative=float(bounded[0, j]), nonnegative_ci95=interval(bounded[1:, j]),
                        nonnegative_at_boundary=bool(bounded[0, j] <= 1e-12),
                        bootstrap_boundary_fraction=float(np.mean(bounded[1:, j] <= 1e-12)))
                    if idx in (0, 2):
                        val['raw_relative_to_residual_energy'] = float(raw[0, j]/energy[0, task])
                        val['raw_relative_ci95'] = interval(raw[1:, j]/energy[1:, task])
                        val['original_no_amplitude_control'] = baseline['components'][name]
                        val['estimate_change_from_original'] = float(raw[0, j]-baseline['components'][name]['raw'])
                    record['components'][name] = val
                    flat.append(dict(scope=scope, task=task_name, model=model, component=name,
                        raw=val['raw'], low=val['raw_ci95'][0], high=val['raw_ci95'][1],
                        nonnegative=val['nonnegative'], boundary_fraction=val['bootstrap_boundary_fraction'],
                        rank=record['rank'], n_components=len(indices), identifiable=record['identifiable'],
                        condition_number=record['condition_number']))
                result[scope][task_name][model] = record
                saved[f'{scope}_{task_name}_{model}_raw'] = raw
                saved[f'{scope}_{task_name}_{model}_nonnegative'] = bounded
    return result, flat, saved


def report(result, original):
    lines = ['# Amplitude-control sensitivity of relation covariance', '',
        'This adds controls to the existing 951D RIDGE_RESPONSE residual-covariance diagnostic. '
        'Original estimates, rows, pair weights and all 1,000 chemical-group node-bootstrap '
        'draws are unchanged. No query diagnostic is used to fit predictive models or gates.', '',
        '## Amplitude kernels', '',
        '- BINS: source log norm is assigned to five bins using each unit\'s saved TRAIN '
        'q20/q40/q60/q80 edges. The kernel is one when the two sources share a bin: a one-hot Gram.',
        '- CONTINUOUS: z = (log norm − (TRAIN q40 + TRAIN q60)/2)/(TRAIN q80 − TRAIN q20). '
        'The kernel is z_i z_j, a rank-one PSD Gram with signed off-diagonal values. '
        'Its diagonal is not one, so its coefficient is not directly an average-coordinate variance.',
        '- BOTH includes both kernels. Each version is reported; none replaces the original specification. '
        'These test two specified amplitude dependencies, not every possible nonlinear amplitude effect.', '',
        '## Pooled target-related coefficients', '',
        '| Task | Model | Target estimate [95% interval] | Signed interaction estimate [95% interval] |',
        '| --- | --- | --- | --- |']
    for task in ('SAME', 'CROSS'):
        for base in ('ADJUSTED', 'SIGNED_JOINT'):
            old = original['scopes']['ALL_SEVEN_PAIRS']['outcomes'][task]['models'][base]
            entries = [(base+' (original)', old)] + [(name, result['scopes']['ALL_SEVEN_PAIRS'][task][name])
                for name in MODELS if name.startswith(base+'_')]
            for name, model in entries:
                values = []
                for component in ('target', 'target_morphology'):
                    c = model['components'].get(component)
                    values.append('—' if c is None else f"{c['raw']:.6g} [{c['raw_ci95'][0]:.6g}, {c['raw_ci95'][1]:.6g}]")
                lines.append(f'| {task} | {name} | {values[0]} | {values[1]} |')
    lines += ['', '## Identifiability', '',
        '| Model | Rank | Components | Condition number | Target residual design fraction | Signed residual design fraction |',
        '| --- | --- | --- | --- | --- | --- |']
    for name, m in result['scopes']['ALL_SEVEN_PAIRS']['SAME'].items():
        lines.append(f"| {name} | {m['rank']} | {m['n_components']} | {m['condition_number']:.5g} | "
            f"{m['target_residual_design_fraction']:.5g} | {m['signed_residual_design_fraction']} |")
    lines += ['', 'Free unit intercepts remain nuisance terms. Kernel coefficients are estimated by '
        'off-diagonal covariance-moment regression, not a full covariance likelihood or a physical '
        'noise decomposition. Raw and NNLS estimates, their intervals and boundary frequencies '
        'are all retained in the JSON/CSV. A coefficient reaching zero is not evidence of no '
        'biological relationship. Uncertainty is conditional on fixed fitted models, folds and '
        'observed batches; no independent verification is supplied.', '']
    return '\n'.join(lines)


def main():
    started = time.monotonic()
    original = json.loads((OUTPUT/'summary.json').read_text())
    saved = read_npz(OUTPUT/'moment_sufficient_statistics.npz', ('gram', 'cross'))
    bootstrap = read_npz(OUTPUT/'bootstrap_components.npz',
                         ('global_groups', 'node_multiplicity', 'residual_energy'))
    assert original['bootstrap_replicates'] == 1000
    with threadpool_limits(limits=2):
        units, groups, _ = load_units()
        np.testing.assert_array_equal(groups, bootstrap['global_groups'])
        for u, old in zip(units, original['units']):
            for key in ('n_row_pairs', 'n_diagnostic_rows', 'n_diagnostic_groups', 'n_target_positive_group_dyads'):
                assert u['info'][key] == old[key]
        amplitude_audit = add_amplitude_design(units)
        g, h = augmented_moments(units, saved['gram'], saved['cross'],
            bootstrap['node_multiplicity'], groups)
        scopes, flat, components = summarize(original, g, h, bootstrap['residual_energy'])
    result = dict(state='COMPLETE', source_diagnostic='variance/summary.json',
        dimension=951, outcome='saved RIDGE_RESPONSE residuals; not CORE measurement noise',
        no_original_results_overwritten=True, exact_original_node_bootstrap_reused=True,
        exact_original_primary_moments_reused=True, same_samples_and_pair_weights=True,
        bootstrap_replicates=1000, seed=original['seed'], n_units=35, n_global_groups=len(groups),
        query_Y_for_diagnostic_only=True, new_predictive_training=False,
        kernel_names=NAMES, model_indices=MODELS, amplitude_kernels_fitted_on='saved TRAIN quintile edges only',
        continuous_scaling='z=(log source norm-(q40+q60)/2)/(q80-q20)',
        amplitude_audit=amplitude_audit, scopes=scopes, elapsed_seconds=time.monotonic()-started)
    write_json(OUTPUT/'amplitude_sensitivity.json', result)
    (OUTPUT/'amplitude_sensitivity.md').write_text(report(result, original))
    with (OUTPUT/'amplitude_sensitivity_components.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    np.savez_compressed(OUTPUT/'amplitude_sensitivity_bootstrap.npz', **components)
    print(json.dumps(dict(state='COMPLETE', seconds=result['elapsed_seconds'],
        output=str(OUTPUT/'amplitude_sensitivity.md'))), flush=True)


if __name__ == '__main__':
    main()
