"""Read completed borrowing artifacts and write a separate structural audit.

Does not fit, sample, select tuning parameters, or modify saved run artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output-name', default='independent_borrowing_results_audit.json')
    args = parser.parse_args()
    root = args.run.resolve()
    summary = json.loads((root / 'summary.json').read_text())
    if summary['state'] != 'COMPLETE':
        raise ValueError('Final audit requires a complete run')
    core = load(root / 'CORE.npz')
    old_core = load(Path(summary['source']) / 'CORE.npz')
    for key in ('ids', 'actual', 'predicted', 'p_null', 'selected', 'crps', 'nll'):
        np.testing.assert_array_equal(core[key], old_core[key])
    ids = core['ids']
    assert len(ids) == len(set(ids)) == summary['n'] == 1188
    index = {v: i for i, v in enumerate(ids)}
    expected_arms = {'CORE', 'CURRENT_GATE'} | {
        f'{channel}_{suffix}' for channel in ('TARGET', 'MOA')
        for suffix in ('A025', 'A050', 'A100', 'FIXED_SELECTED', 'DELTA_RIDGE', 'DELTA_BOOST')}
    actual_arms = {p.stem for p in root.glob('*.npz')}
    assert actual_arms == expected_arms
    arrays = {name: load(root / f'{name}.npz') for name in expected_arms}
    union = core['support_target'] | core['support_moa']
    metric_keys = ('predicted', 'p_null', 'crps', 'energy', 'nll', 'brier',
                   'joint_coverage_by_level', 'covariance_u')
    reports = {}
    for arm, out in arrays.items():
        np.testing.assert_array_equal(out['ids'], ids)
        np.testing.assert_array_equal(out['actual'], core['actual'])
        channel = arm.split('_')[0]
        support = core['support_' + channel.lower()] if channel in ('TARGET', 'MOA') else union
        np.testing.assert_array_equal(out['resource_support'], support)
        coefficient = out['coefficients']
        assert coefficient.shape == (len(ids), 3)
        assert (coefficient >= 0).all()
        np.testing.assert_allclose(coefficient.sum(1), 1, atol=1e-14, rtol=0)
        active = out['effective_mixing'] > 0
        np.testing.assert_allclose(out['effective_mixing'], coefficient[:, 1:].sum(1), atol=1e-14)
        assert not np.any(active & ~support)
        for key in metric_keys:
            np.testing.assert_array_equal(out[key][~active], core[key][~active])
        np.testing.assert_allclose(out['brier'], (out['p_null'] - (out['actual'] <= 0)) ** 2,
                                   atol=0, rtol=0)
        membership_changes = out['selected'] != core['selected']
        selected_mask = out['selected'].astype(bool)
        reports[arm] = dict(supported=int(support.sum()), active=int(active.sum()),
            selected=int(selected_mask.sum()), selected_null=int((out['actual'][selected_mask] <= 0).sum()),
            changed_membership=int(membership_changes.sum()),
            unsupported_membership_spillovers=int((membership_changes & ~support).sum()),
            exact_off=True)
        metric = summary['metrics'][arm]
        assert metric['selected_n'] == int(selected_mask.sum())
        assert metric['selected_null'] == int((out['actual'][selected_mask] <= 0).sum())
        np.testing.assert_allclose(metric['selected_mean'], out['actual'][selected_mask].mean(), atol=0, rtol=0)
        assert metric['changed_selected_membership'] == int(membership_changes.sum())
        assert metric['active_n'] == int(active.sum())
        assert metric['support_n'] == int(support.sum())
        for subset, mask in (('full', np.ones(len(ids), bool)), ('supported', support),
                             ('union_supported', union), ('active', active)):
            saved_metric = metric[subset]
            assert saved_metric['n'] == int(mask.sum())
            if not mask.any():
                continue
            for key, value in saved_metric['scores'].items():
                np.testing.assert_allclose(value, out[key][mask].mean(), atol=0, rtol=0)
            np.testing.assert_allclose(saved_metric['actual_null_rate'],
                                       (out['actual'][mask] <= 0).mean(), atol=0, rtol=0)
            np.testing.assert_allclose(saved_metric['p_null_mean'], out['p_null'][mask].mean(), atol=0, rtol=0)
            for j, value in enumerate(saved_metric['joint_coverage'].values()):
                np.testing.assert_allclose(value, out['joint_coverage_by_level'][mask, j].mean(), atol=0, rtol=0)
    cell_reports, seen = [], np.zeros(len(ids), int)
    for cell in summary['cells']:
        folder = root / f"cell_{cell['fold']}_{cell['half']}"
        q = np.array([index[v] for v in cell['query_ids']])
        seen[q] += 1
        plans = load(folder / 'reference_plans.npz')
        np.testing.assert_array_equal(plans['query_ids'], ids[q])
        np.testing.assert_array_equal(plans['reference_ids'], cell['calibration_ids'])
        assert not set(cell['query_ids']) & set(cell['calibration_ids'])
        assert not set(cell['model_fit_ids']) & set(cell['query_ids'] + cell['calibration_ids'])
        groups = core['groups']
        group_sets = [set(groups[[index[v] for v in cell[k]]])
                      for k in ('query_ids', 'calibration_ids', 'model_fit_ids')]
        assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
        for arm, out in arrays.items():
            local = load(folder / f'{arm}.npz')
            for key in local:
                np.testing.assert_array_equal(local[key], out[key][q])
            np.testing.assert_array_equal(plans[f'{arm}_coefficients'], out['coefficients'][q])
            selected = np.zeros(len(q), bool)
            score = out['predicted'][q] - .2 * out['p_null'][q]
            selected[np.lexsort((ids[q], -score))[:cell['budget']]] = True
            np.testing.assert_array_equal(selected, out['selected'][q])
        cell_reports.append(dict(fold=cell['fold'], half=cell['half'], queries=len(q),
                                 reference_groups=len(group_sets[1]), budget=cell['budget']))
    np.testing.assert_array_equal(seen, np.ones(len(ids), int))
    # Independently reconstruct each saved mixture from its three endpoints.
    endpoint_arms = ('CORE', 'TARGET_A100', 'MOA_A100')
    endpoint_mean = np.stack([arrays[a]['predicted'] for a in endpoint_arms], axis=1)
    endpoint_prob = np.stack([arrays[a]['p_null'] for a in endpoint_arms], axis=1)
    endpoint_nll = np.stack([arrays[a]['nll'] for a in endpoint_arms], axis=1)
    maxima = dict(mean_affine=0., probability_affine=0., density_mixture=0., score_quadratic=0.)
    for arm, out in arrays.items():
        c = out['coefficients']
        for key, endpoints, label in (('predicted', endpoint_mean, 'mean_affine'),
                                      ('p_null', endpoint_prob, 'probability_affine')):
            expected = np.sum(c * endpoints, axis=1)
            maxima[label] = max(maxima[label], float(np.max(np.abs(out[key] - expected))))
            np.testing.assert_allclose(out[key], expected, atol=2e-12, rtol=2e-12)
        logc = np.full_like(c, -np.inf)
        np.log(c, out=logc, where=c > 0)
        expected_nll = -logsumexp(logc - endpoint_nll, axis=1)
        maxima['density_mixture'] = max(maxima['density_mixture'], float(np.max(np.abs(out['nll'] - expected_nll))))
        np.testing.assert_allclose(out['nll'], expected_nll, atol=2e-10, rtol=2e-12)
    for channel in ('TARGET', 'MOA'):
        for key in ('crps', 'energy', 'single_crps', 'pair_crps', 'average_crps', 'absolute_pair_crps'):
            expected = .375 * core[key] + .75 * arrays[channel + '_A050'][key] - .125 * arrays[channel + '_A100'][key]
            observed = arrays[channel + '_A025'][key]
            maxima['score_quadratic'] = max(maxima['score_quadratic'], float(np.max(np.abs(expected - observed))))
            np.testing.assert_allclose(observed, expected, atol=2e-10, rtol=2e-12)
    provenance = {}
    previous = summary.get('completed_cells_reused_from')
    if previous:
        previous = Path(previous)
        for name in ('PROTOCOL.md', 'biology_delta_gate.py', 'module_comparison_audit.py',
                     'BIOLOGY_BORROWING_LKCP_APPLICABILITY.json'):
            assert (root / name).read_bytes() == (previous / name).read_bytes()
        for folder in sorted(previous.glob('cell_*')):
            if not (folder / 'cell.json').exists():
                continue
            files = [f for f in folder.iterdir() if f.is_file()]
            assert all(f.read_bytes() == (root / folder.name / f.name).read_bytes() for f in files)
            provenance[folder.name] = dict(files=len(files), byte_identical=True)
    report = dict(state='PASS', run=str(root), n=len(ids), cells=len(cell_reports),
        prediction_and_mixture_checks=reports, cells_checked=cell_reports,
        maximum_numerical_discrepancies=maxima, reused_cells=provenance,
        audit_scope='Saved-data structural and arithmetic audit; no model refit or query-based rule selection',
        formal_certificate=False,
        interpretation='Supported subgroup increments are descriptive, not causal ATT. Fixed-budget selection can change membership of unchanged unsupported queries.')
    output = root / args.output_name
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(state='PASS', output=str(output), checks=maxima, reused=list(provenance)), indent=2))


if __name__ == '__main__':
    main()
