"""Fixed biological borrowing and nested delta gates on opened LINCS DEV.

No world-model fit, endpoint change, query-based gate selection or new data.
The complete amplitude-conditioned CORE is the reference for every comparison.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .biology_delta_gate import fit_biology_delta_gates
from .biology_kernel_evaluation import write_json
from .conditional_joint_error_experiment import observable_forward
from .empirical_radial import fit_radial
from .empirical_radial_experiment import LEVELS
from .frozen_acquisition_policy import select_frozen_cohort_plan
from .gram_geometry import profiles_to_gram, gram_to_coordinates
from .lincs_biology_experiment import load_data
from .module_comparison_audit import audit_paired_module
from .module_switch_experiment import biology_similarity
from .optional_radial_modules import supported_retrieval, fit_radial_switch, BIO_COEFFICIENTS
from .radial_mixture_evaluation import evaluate_radial_mixtures
from .reference_information_diagnostic import bootstrap_difference

PROJECT = Path(__file__).resolve().parents[1]
SEED = 20260916
SCORES = ('crps', 'brier', 'nll', 'energy', 'single_crps', 'pair_crps',
          'average_crps', 'absolute_pair_crps')
PRIMARY = ('crps', 'brier', 'nll', 'energy')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def arm_support(name, target, moa):
    if name.startswith('TARGET_'):
        return target
    if name.startswith('MOA_'):
        return moa
    return target | moa


def scalar_metrics(out, actual, mask):
    take = np.asarray(mask, bool)
    if not take.any():
        return dict(n=0)
    truth = actual[take] <= 0
    return dict(n=int(take.sum()), actual_null_rate=float(truth.mean()),
        p_null_mean=float(out['p_null'][take].mean()),
        null_auc=float(roc_auc_score(truth, out['p_null'][take])) if len(np.unique(truth)) == 2 else None,
        gamma_spearman=float(spearmanr(actual[take], out['predicted'][take]).statistic),
        scores={key: float(out[key][take].mean()) for key in SCORES},
        joint_coverage={str(level): float(out['joint_coverage_by_level'][take, j].mean())
                        for j, level in enumerate(LEVELS)})


def paired_intervals(out, core, mask, groups, layouts, *, keys=PRIMARY):
    mask = np.asarray(mask, bool)
    if not mask.any():
        return dict(n=0)
    result = dict(n=int(mask.sum()), chemistry_groups=len(np.unique(groups[mask])),
                  layouts=len(np.unique(layouts[mask])))
    for key in keys:
        result[key] = {label: bootstrap_difference(out[key][mask], core[key][mask], labels[mask])
                      for label, labels in (('chemistry', groups), ('layout', layouts))}
    return result


def aggregate(stores, ids, actual, groups, layouts, cell_ids, target, moa, cells):
    core = stores['CORE']
    all_rows = np.ones(len(ids), bool)
    union = target | moa
    metrics, comparisons, audits = {}, {}, {}
    for arm, out in stores.items():
        out['brier'] = (out['p_null']-(actual <= 0))**2
        out['policy_value'] = out['selected']*actual
        out['policy_null'] = out['selected']*(actual <= 0)
    for arm, out in stores.items():
        support = arm_support(arm, target, moa)
        selected = out['selected'].astype(bool)
        active = out['effective_mixing'] > 0
        metrics[arm] = dict(full=scalar_metrics(out, actual, all_rows),
            supported=scalar_metrics(out, actual, support),
            union_supported=scalar_metrics(out, actual, union),
            active=scalar_metrics(out, actual, active),
            support_n=int(support.sum()), active_n=int(active.sum()),
            mean_mixing_supported=float(out['effective_mixing'][support].mean()) if support.any() else 0.,
            selected_n=int(selected.sum()), selected_null=int((actual[selected] <= 0).sum()),
            selected_mean=float(actual[selected].mean()),
            changed_selected_membership=int(np.count_nonzero(selected != core['selected'])),
            by_cell=[dict(cell=int(c), n=int((cell_ids == c).sum()),
                crps=float(out['crps'][cell_ids == c].mean()),
                brier=float(out['brier'][cell_ids == c].mean())) for c in np.unique(cell_ids)])
        if arm == 'CORE':
            continue
        comparisons[arm] = dict(
            supported=paired_intervals(out, core, support, groups, layouts),
            union_supported=paired_intervals(out, core, union, groups, layouts),
            full=paired_intervals(out, core, all_rows, groups, layouts),
            allocation=paired_intervals(out, core, all_rows, groups, layouts,
                                        keys=('policy_value', 'policy_null')))
        if 'DELTA_' in arm:
            fixed = stores[arm.split('_')[0]+'_FIXED_SELECTED']
            comparisons[arm]['vs_calibration_selected_fixed'] = paired_intervals(
                out, fixed, support, groups, layouts)
        audit_keys = ('predicted', 'p_null', 'crps', 'nll', 'brier', 'joint_coverage_by_level')
        audits[arm] = audit_paired_module(ids=ids,
            core_predictions={key: core[key] for key in audit_keys},
            module_predictions={key: out[key] for key in audit_keys},
            eligible_support=support, active_mask=active,
            core_selected=core['selected'].astype(bool), module_selected=selected,
            actual_gamma=actual, cell_ids=cell_ids)
    lookup = {v:i for i,v in enumerate(ids)}
    lambda0 = np.zeros(len(ids), bool)
    random_total, random_null = 0., 0.
    for cell in cells:
        q = np.array([lookup[v] for v in cell['query_ids']]); k = cell['budget']
        order = np.lexsort((ids[q], -core['predicted'][q]))
        lambda0[q[order[:k]]] = True
        random_total += k*float(actual[q].mean())
        random_null += k*float((actual[q] <= 0).mean())
    budget = int(core['selected'].sum())
    references = dict(core_lambda0=dict(selected_n=int(lambda0.sum()),
        selected_null=int((actual[lambda0] <= 0).sum()), selected_mean=float(actual[lambda0].mean())),
        uniform_random_expectation=dict(selected_n=budget, selected_null=random_null,
                                       selected_mean=random_total/budget))
    return metrics, comparisons, audits, references


def run(source, output, *, reuse_completed=None):
    started = time.monotonic()
    source, root = Path(source).resolve(), Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    previous = read_json(source/'summary.json')
    radial_source = Path(previous['source'])
    old = read_json(radial_source/'summary.json')
    manifest = read_json(Path(old['reference_run'])/'run_manifest.json')
    data, metadata = load_data(old['data_directory'])
    ids, groups = data['ids'], data['groups']
    n = len(ids)
    if n != 1188 or ids.tolist() != manifest['ids']:
        raise ValueError('Only the declared, already-opened LINCS population is allowed')
    core_saved = read_npz(source/'CORE.npz')
    prior = read_npz(radial_source/'AMP_EMP_LOCAL.npz')
    np.testing.assert_array_equal(core_saved['ids'], ids)
    np.testing.assert_array_equal(prior['ids'], ids)
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/BIOLOGY_BORROWING_DIAGNOSTIC_PLAN_20260916.md', root/'PROTOCOL.md')
    for filename in ('biology_borrowing_experiment.py', 'biology_delta_gate.py',
                     'radial_mixture_evaluation.py', 'module_comparison_audit.py'):
        shutil.copy2(PROJECT/'opal2'/filename, root/filename)
    applicability = source.parent/'BIOLOGY_BORROWING_LKCP_APPLICABILITY.json'
    shutil.copy2(applicability, root/applicability.name)
    layouts = np.array([unit['layout_block'] for unit in metadata['units']])
    raw = gram_to_coordinates(profiles_to_gram(torch.tensor(data['Y']))).numpy()
    actual, observed, difference, _ = observable_forward(raw)
    np.testing.assert_allclose(actual, core_saved['actual'], atol=1e-12, rtol=1e-12)
    amp = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    norm2 = np.square(data['Y'][:, 0]).mean(1)
    absolute = np.log1p(difference*norm2[:, None])
    lookup = {v:i for i,v in enumerate(ids)}
    records = {record['fold']: record for record in manifest['folds']}
    stores, cells = {}, []
    seen = np.zeros(n, int)
    all_target, all_moa = np.zeros(n, bool), np.zeros(n, bool)
    cell_ids = np.full(n, -1, int)
    for cell_number, cell in enumerate(old['cells']):
        fold, half = cell['fold'], cell['half']
        folder = root/f'cell_{fold}_{half}'
        q = np.array([lookup[v] for v in cell['query_ids']])
        cal = np.array([lookup[v] for v in cell['representative_ids']])
        fit = np.asarray(records[fold]['fit'], int)
        if set(groups[fit]) & set(groups[np.r_[q, cal]]) or set(groups[q]) & set(groups[cal]):
            raise ValueError('QUERY or CAL entered MODEL_FIT/reference fitting')
        previous_cell = Path(reuse_completed)/folder.name if reuse_completed else None
        if previous_cell is not None and (previous_cell/'cell.json').exists():
            saved_record = read_json(previous_cell/'cell.json')
            if (saved_record['query_ids'] != ids[q].tolist()
                    or saved_record['calibration_ids'] != ids[cal].tolist()
                    or saved_record['model_fit_ids'] != ids[fit].tolist()
                    or saved_record['budget'] != cell['budget']):
                raise ValueError('Completed cell does not match the frozen roles')
            arm_files = [f for f in previous_cell.glob('*.npz') if f.name != 'reference_plans.npz']
            if len(arm_files) != 14:
                raise ValueError('Completed cell must contain every prespecified arm')
            for file in arm_files:
                out = read_npz(file)
                np.testing.assert_array_equal(out.pop('ids'), ids[q])
                np.testing.assert_array_equal(out.pop('actual'), actual[q])
                arm = file.stem
                if arm == 'CORE':
                    for key in ('predicted', 'p_null', 'selected', 'crps', 'nll'):
                        np.testing.assert_array_equal(out[key], core_saved[key][q])
                stores.setdefault(arm, {})
                for key, value in out.items():
                    if key not in stores[arm]:
                        stores[arm][key] = np.empty((n, *value.shape[1:]), dtype=value.dtype)
                    stores[arm][key][q] = value
            all_target[q] = stores['TARGET_A025']['resource_support'][q]
            all_moa[q] = stores['MOA_A025']['resource_support'][q]
            shutil.copytree(previous_cell, folder)
            cells.append(saved_record); seen[q] += 1; cell_ids[q] = cell_number
            print(f'cell={cell_number+1}/10 reused complete, unchanged artifact', flush=True)
            continue
        folder.mkdir()
        ref = read_npz(radial_source/f'cell_{fold}_{half}_radial.npz')
        np.testing.assert_array_equal(ref['cal_ids'], ids[cal])
        np.testing.assert_array_equal(ref['query_ids'], ids[q])
        base, radii = ref['local_weights'], ref['amplitude_radii']
        similarities = biology_similarity(data, metadata, q, cal)
        cal_similarities = biology_similarity(data, metadata, cal, cal)
        channels = [supported_retrieval(s) for s in similarities]
        target, moa = (c['supported'] for c in channels)
        all_target[q], all_moa[q] = target, moa
        endpoints = {'CORE': base.copy()}
        coefficients = {'CORE': np.tile([1., 0., 0.], (len(q), 1))}
        for j, (relation, channel) in enumerate(zip(('TARGET', 'MOA'), channels), 1):
            weights = channel['weights'].copy()
            weights[~channel['supported']] = base[~channel['supported']]
            endpoints[relation] = weights
            for alpha, label in ((.25, '025'), (.5, '050'), (1., '100')):
                amount = alpha*channel['supported']
                coef = np.zeros((len(q), 3)); coef[:, 0] = 1-amount; coef[:, j] = amount
                coefficients[relation+'_A'+label] = coef
        original_gate = fit_radial_switch(radii, amp[cal], groups[cal], cal_similarities,
            fit_amp_sd=cell['local_bandwidth'], coefficient_grid=BIO_COEFFICIENTS)
        current = np.zeros((len(q), 3))
        for j, (alpha, channel) in enumerate(zip(original_gate.coefficients, channels), 1):
            current[:, j] = alpha*channel['gate']
        current[:, 0] = 1-current[:, 1:].sum(1)
        coefficients['CURRENT_GATE'] = current
        write_json(root/'status.json', dict(state='RUNNING', phase='nested_delta_fitting',
            cell=cell_number, cells_complete=len(cells), elapsed_seconds=time.monotonic()-started))
        gate = fit_biology_delta_gates(radii, amp[cal], groups[cal],
            dict(zip(('target', 'moa'), cal_similarities)), ids[cal],
            fit_amp_sd=cell['local_bandwidth'], seed=SEED+cell_number*100)
        plans = gate.apply(amp[q], groups[q], dict(zip(('target', 'moa'), similarities)), ids[q],
                           base_weights=base)['plans']
        gate.save(folder/'delta_gates.joblib')
        write_json(folder/'gate_development.json', gate.report)
        for name, plan in plans.items():
            j = 1 if name.startswith('TARGET_') else 2
            coef = np.zeros((len(q), 3)); coef[:, 0] = 1-plan['alpha']; coef[:, j] = plan['alpha']
            coefficients[name] = coef
            mixed = sum(coef[:, k, None]*w for k, w in enumerate(endpoints.values()))
            np.testing.assert_allclose(mixed, plan['weights'], atol=2e-15, rtol=0)
        write_json(root/'status.json', dict(state='RUNNING', phase='full_joint_sampling',
            cell=cell_number, cells_complete=len(cells), elapsed_seconds=time.monotonic()-started))
        stats = read_json(Path(old['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json')
        scores = evaluate_radial_mixtures(prior['mean_u'][q], prior['scatter_u'][q],
            prior['actual_u'][q], stats, actual[q], observed[q], absolute[q], norm2[q],
            cell['normal_seed'], law=fit_radial(radii), endpoint_weights=endpoints,
            coefficients=coefficients, samples=100000,
            baseline_saved={k:v[q] for k,v in core_saved.items() if isinstance(v, np.ndarray) and v.shape[:1] == (n,)})
        for arm, out in scores.items():
            amount = coefficients[arm][:, 1:].sum(1)
            support = arm_support(arm, target, moa)
            if np.any((amount > 0) & ~support):
                raise ValueError('Unsupported biological activation')
            choice = select_frozen_cohort_plan(ids[q], out['predicted'], out['p_null'], cell['budget'])
            out.update(selected=np.asarray(choice.selected_mask, int), effective_mixing=amount,
                       resource_support=support, coefficients=coefficients[arm])
            if arm == 'CORE':
                for key in ('predicted', 'p_null', 'selected', 'crps', 'nll'):
                    np.testing.assert_array_equal(out[key], core_saved[key][q])
            stores.setdefault(arm, {})
            for key, value in out.items():
                if key not in stores[arm]:
                    stores[arm][key] = np.empty((n, *value.shape[1:]), dtype=value.dtype)
                stores[arm][key][q] = value
            np.savez_compressed(folder/(arm+'.npz'), ids=ids[q], actual=actual[q], **out)
        cell_record = dict(fold=fold, half=half, query_ids=ids[q].tolist(), budget=cell['budget'],
            calibration_ids=ids[cal].tolist(), model_fit_ids=ids[fit].tolist(),
            current_gate=original_gate.selection, current_coefficients=original_gate.coefficients,
            support={relation: {k:v for k,v in channel.items() if k != 'weights'}
                     for relation, channel in zip(('target', 'moa'), channels)},
            final_fixed_alphas={name: gate.report['channels'][name]['final_fixed_alpha'] for name in ('target', 'moa')})
        write_json(folder/'cell.json', cell_record)
        np.savez_compressed(folder/'reference_plans.npz', query_ids=ids[q], reference_ids=ids[cal],
            **{'endpoint_'+k:v for k,v in endpoints.items()},
            **{k+'_coefficients':v for k,v in coefficients.items()},
            **{k+'_predicted_delta':v['predicted_delta'] for k,v in plans.items()})
        cells.append(cell_record); seen[q] += 1; cell_ids[q] = cell_number
        elapsed = time.monotonic()-started
        write_json(root/'status.json', dict(state='RUNNING', phase='cells', cells_complete=len(cells),
                                          elapsed_seconds=elapsed))
        print(f'cell={cell_number+1}/10 support_target={target.sum()} support_moa={moa.sum()} elapsed={elapsed:.1f}s', flush=True)
    np.testing.assert_array_equal(seen, np.ones(n, int))
    metrics, comparisons, audits, references = aggregate(stores, ids, actual, groups,
        layouts, cell_ids, all_target, all_moa, cells)
    for arm, out in stores.items():
        np.savez_compressed(root/(arm+'.npz'), ids=ids, groups=groups, layout=layouts,
            cell=cell_ids, actual=actual, support_target=all_target, support_moa=all_moa, **out)
    summary = dict(state='COMPLETE', n=n, samples=100000, source=str(source), cells=cells,
        metrics=metrics, comparisons=comparisons, references=references,
        support_counts=dict(target=int(all_target.sum()), moa=int(all_moa.sum()), union=int((all_target|all_moa).sum())),
        elapsed_seconds=time.monotonic()-started, world_model_retrained=False, endpoint_changed=False,
        formal_certificate=False, query_gate_selection=False, protocol=str(root/'PROTOCOL.md'),
        scope='Opened LINCS development; primary supported-subgroup distribution comparison; full-cohort frozen-budget policy',
        uncertainty='Paired group/layout resampling conditional on fitted rules, without gate-refitting or multiple-comparison guarantee',
        reference_cost_included=False, lkcp_applicability=read_json(root/applicability.name),
        completed_cells_reused_from=str(reuse_completed) if reuse_completed else None,
        representations_tested=False)
    write_json(root/'summary.json', summary)
    write_json(root/'paired_module_audit.json', audits)
    write_json(root/'status.json', dict(state='COMPLETE', cells_complete=len(cells), elapsed_seconds=summary['elapsed_seconds']))
    print(json.dumps(dict(state='COMPLETE', output=str(root), elapsed_seconds=summary['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=str(PROJECT/'runs/module_switches_20260916_v1/lincs_v2'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--reuse-completed', help='Completed identical cells from an interrupted run; no outcomes used for selection')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        run(args.source, args.output, reuse_completed=args.reuse_completed)


if __name__ == '__main__':
    main()
