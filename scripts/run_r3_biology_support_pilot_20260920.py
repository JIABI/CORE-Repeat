"""Small, fixed-design support-mask pilot; original CORE/R3 files are read-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from opal2.biology_kernel_evaluation import write_json
from opal2.dual_branch_features import apply_increment
from opal2.empirical_radial_experiment import score
from opal2.eu_core_experiment import extra_seed_moments, select
from opal2.eu_r3_biology import build_eu_biology_features
from opal2.gram_oof_ridge import transform_target
from opal2.reference_information_diagnostic import bootstrap_difference
from opal2.rxrx3_r3_cache import load_cell
from opal2.support_gated_biology import fit_right as fit_modified_right
from scripts.run_r3_rxrx3_modules_20260920 import enriched_scope, CORE_SEED, SEED
from scripts.run_r2_core_comparison_20260917 import read_npz

SOURCE = PROJECT/'runs/r3_rxrx3_modules_20260920_v1'
ROOT = PROJECT/'runs/r3_biology_support_pilot_20260920_v1'
CELLS = (0, 3, 5, 7)
SAMPLES = 100000
OFFSETS = (100000, 200000)
BASELINES = ('CORE', 'GELU', 'BIO_STRUCTURED', 'BIO_RANDOM')
NEW_ARMS = ('MASK_ONLY_REAL', 'MASK_ONLY_RANDOM',
            'MASK_CONFIDENCE_REAL', 'MASK_CONFIDENCE_RANDOM')
ARMS = BASELINES + NEW_ARMS
METRICS = ('nll', 'energy', 'crps', 'brier', 'single_crps', 'pair_crps',
           'average_crps', 'absolute_pair_crps')


def run_cell(index, scope):
    folder, source = ROOT/f'fold_{index}', SOURCE/f'fold_{index}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        return
    started = time.monotonic()
    def status(stage, **extra):
        write_json(folder/'status.json', dict(state='RUNNING', stage=stage,
            cell=index, elapsed_seconds=time.monotonic()-started, **extra))
        print(f'cell={index} {stage} elapsed={time.monotonic()-started:.1f}s {extra}', flush=True)
    status('load frozen data and left branch')
    cell = load_cell(f'fold_{index}', scope=scope)
    data, part, means, stats, arrays, state = (cell[k] for k in
        ('data', 'part', 'means', 'stats', 'oldarrays', 'state'))
    t, r, c, q = (part[k] for k in ('TRAIN', 'REF_FIT', 'DIST_CAL', 'DEV_EVAL'))
    meta = dict(scope['biology_metadata'])
    meta['units'] = [scope['biology_metadata']['units'][int(i)] for i in cell['global_rows']]
    old_predictions = read_npz(source/'module_predictions.npz')
    records = read_npz(source/'adapter_reference/records.npz')
    np.testing.assert_array_equal(records['ids'], data['ids'][r])
    np.testing.assert_array_equal(old_predictions['query_ids'], data['ids'][q])
    left = joblib.load(source/'GELU.joblib')
    transform = joblib.load(source/'descriptors.joblib')
    desc = transform.transform(data, meta).values
    left_q = left.predict_components(desc[q])['total']
    np.testing.assert_array_equal(left_q, old_predictions['GELU_query'])
    scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
    rawmean = means*scale+center
    raw, actual, observed, difference = (cell[k] for k in ('raw', 'actual', 'observed', 'difference'))
    logamp = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    qcov = arrays['query_scatter_u']*arrays['radial_variance_multiplier'][:, None, None]
    random_q = build_eu_biology_features(data, meta, q, r, rawmean[q],
        qcov*scale[None, :, None]*scale[None, None, :], raw[r]-rawmean[r],
        amplitude_edges=np.quantile(logamp[t], [.2, .4, .6, .8]), random_seed=SEED+100*index+7000)
    held = {'REAL': dict(values=old_predictions['query_biology'], support=old_predictions['query_support']),
            'RANDOM': random_q}
    names = json.loads((source/'adapter_reference/summary.json').read_text())['biological_names']
    # Verify the recreated random inputs reproduce the exact original carrier.
    random_old = joblib.load(source/'BIO_RANDOM.joblib')
    np.testing.assert_allclose(random_old.predict_components(desc[q], random_q['values'],
        random_q['support'])['total'], old_predictions['BIO_RANDOM_query'], rtol=1e-12, atol=1e-12)
    target = transform_target(raw[q], stats)
    norm2 = np.square(np.asarray(data['Y'][q, 0], dtype=float)).mean(1)
    absolute = np.log1p(difference[q]*norm2[:, None])
    baseline = read_npz(source/'GELU.npz')
    fitting = {}
    for arm in NEW_ARMS:
        gate_mode, relation = arm.rsplit('_', 1)
        prefix = 'biology' if relation == 'REAL' else 'random_biology'
        b = held[relation]
        checkpoint = folder/(arm+'.joblib')
        status('fit right only', arm=arm)
        if checkpoint.exists():
            model = joblib.load(checkpoint)
        else:
            model = fit_modified_right(left, desc[r], records[prefix+'_values'], names,
                records[prefix+'_support'], records['energies'], data['ids'][r],
                mode='structured', gate_mode=gate_mode, seed=SEED+100*index+100)
            joblib.dump(model, checkpoint)
        write_json(folder/(arm+'_training.json'), model.report)
        comp = model.predict_components(desc[q], b['values'], b['support'])
        np.testing.assert_array_equal(comp['left'], left_q)
        np.testing.assert_array_equal(model.predict_components(desc[q], b['values'], b['support'],
            right_enabled=False)['total'], left_q)
        np.testing.assert_array_equal(model.predict_components(desc[q], b['values'], b['support'],
            enabled=False)['total'], np.zeros_like(left_q))
        unsupported = ~b['support']
        np.testing.assert_array_equal(comp['total'][unsupported], left_q[unsupported])
        increment = comp['total']
        scatter = apply_increment(rawmean[q], scale, arrays['query_scatter_u'], increment)
        path = folder/(arm+'.npz')
        status('100k full distribution scores', arm=arm)
        if path.exists():
            saved = read_npz(path)
            np.testing.assert_array_equal(saved['increment'], increment)
        else:
            out = score(means[q], scatter, target, stats, actual[q], observed[q], absolute,
                norm2, CORE_SEED+100*index, law=state['law'],
                weights=arrays['radial_weights'], samples=SAMPLES)
            for key in ('predicted', 'p_null', 'crps', 'nll', 'energy'):
                np.testing.assert_array_equal(out[key][unsupported], baseline[key][unsupported])
            out.update(mean_u=means[q], actual_u=target, scatter_u=scatter,
                increment=increment, right_increment=comp['right'], resource_support=b['support'],
                brier=np.square(out['p_null']-(actual[q]<=0)))
            for lam in (.2, 0.):
                out[f'selected_lambda_{lam:g}'] = select(data['ids'][q], out['predicted'], out['p_null'], lam)
            np.savez_compressed(path, ids=data['ids'][q], groups=data['groups'][q], actual=actual[q], **out)
        for offset in OFFSETS:
            dest = folder/f'{arm}_mc{offset}.npz'
            if not dest.exists():
                status('additional-seed decision moments', arm=arm, offset=offset)
                moment = extra_seed_moments(means[q], scatter, stats, law=state['law'],
                    weights=arrays['radial_weights'], seed=CORE_SEED+100*index+offset)
                old_mc = read_npz(source/f'GELU_mc{offset}.npz')
                for key in moment:
                    np.testing.assert_array_equal(moment[key][unsupported], old_mc[key][unsupported])
                np.savez_compressed(dest, ids=data['ids'][q], **moment)
        fitting[arm] = dict(supported_training=model.report['fitting_supported_n'],
            query_right_rms=float(np.sqrt(np.mean(comp['right'][b['support']]**2))),
            n_supported=int(b['support'].sum()))
    np.savez_compressed(folder/'population.npz', ids=data['ids'][q], groups=data['groups'][q],
        layout=data['layout'][q], actual=actual[q], support=old_predictions['query_support'])
    write_json(folder/'complete.json', dict(cell=index, dose=cell['cell_info']['dose_uM'],
        n=len(q), supported=int(old_predictions['query_support'].sum()), fitting=fitting,
        elapsed_seconds=time.monotonic()-started, CORE_retrained=False, left_retrained=False))
    write_json(folder/'status.json', dict(state='COMPLETE', cell=index,
        elapsed_seconds=time.monotonic()-started))


def summarize():
    population = [read_npz(ROOT/f'fold_{i}/population.npz') for i in CELLS]
    pop = {k: np.concatenate([p[k] for p in population]) for k in population[0]}
    out, by_cell, mc = {}, {}, {}
    for arm in ARMS:
        root = SOURCE if arm in BASELINES else ROOT
        cells = [read_npz(root/f'fold_{i}'/(arm+'.npz')) for i in CELLS]
        for p, c in zip(population, cells):
            np.testing.assert_array_equal(p['ids'], c['ids'])
        fields = METRICS+('predicted', 'p_null', 'selected_lambda_0.2', 'joint_coverage_by_level',
                         'gamma_coverage_by_level', 'coordinate_coverage_by_level')
        out[arm] = {k: np.concatenate([c[k] for c in cells]) for k in fields}
        mc[arm] = []
        for offset in (0,)+OFFSETS:
            chosen, predicted = [], []
            for i, p in zip(CELLS, population):
                suffix = '' if offset == 0 else f'_mc{offset}'
                a = read_npz(root/f'fold_{i}'/(arm+suffix+'.npz'))
                chosen.append(select(p['ids'], a['predicted'], a['p_null'], .2))
                predicted.append(a['p_null'])
            s = np.concatenate(chosen)
            pp = np.concatenate(predicted)
            mc[arm].append(dict(offset=offset, selected=int(s.sum()),
                null=int(np.sum(s & (pop['actual']<=0))), total_value=float(pop['actual'][s].sum()),
                expected_null=float(pp[s].sum()),
                list_changes_vs_primary=int(np.sum(s != out[arm]['selected_lambda_0.2']))))
        by_cell[arm] = [dict(cell=i, **{k:float(c[k].mean()) for k in METRICS},
            selected_null=int(np.sum(c['selected_lambda_0.2'] & (p['actual']<=0))))
            for i, c, p in zip(CELLS, cells, population)]
    metrics = {}
    for arm, a in out.items():
        metrics[arm] = {}
        for scope, mask in (('all', np.ones(len(pop['ids']), bool)), ('supported', pop['support'])):
            metrics[arm][scope] = {k:float(a[k][mask].mean()) for k in METRICS}
            for k in ('joint_coverage_by_level', 'gamma_coverage_by_level', 'coordinate_coverage_by_level'):
                metrics[arm][scope][k] = a[k][mask].mean(0)
        selected = a['selected_lambda_0.2']
        metrics[arm]['decision'] = dict(mc[arm][0], mean_value=float(pop['actual'][selected].mean()),
            selected_overlap_CORE=int(np.sum(selected & out['CORE']['selected_lambda_0.2'])),
            selected_overlap_GELU=int(np.sum(selected & out['GELU']['selected_lambda_0.2'])))
    pairs = [('MASK_ONLY_REAL', 'BIO_STRUCTURED'), ('MASK_ONLY_RANDOM', 'BIO_RANDOM'),
        ('MASK_CONFIDENCE_REAL', 'MASK_ONLY_REAL'), ('MASK_CONFIDENCE_RANDOM', 'MASK_ONLY_RANDOM'),
        ('MASK_ONLY_REAL', 'MASK_ONLY_RANDOM'), ('MASK_CONFIDENCE_REAL', 'MASK_CONFIDENCE_RANDOM'),
        ('MASK_ONLY_REAL', 'GELU'), ('MASK_CONFIDENCE_REAL', 'GELU'),
        ('MASK_CONFIDENCE_REAL', 'CORE')]
    comparisons = {}
    for a, b in pairs:
        key = a+' minus '+b
        comparisons[key] = {}
        for scope, mask in (('all', np.ones(len(pop['ids']), bool)), ('supported', pop['support'])):
            comparisons[key][scope] = {k:bootstrap_difference(out[a][k][mask], out[b][k][mask],
                pop['groups'][mask], seed=20260920) for k in METRICS}
        sa, sb = (out[arm]['selected_lambda_0.2'] for arm in (a, b))
        comparisons[key]['value_per_candidate'] = bootstrap_difference(pop['actual']*sa,
            pop['actual']*sb, pop['groups'], seed=20260920)
    result = dict(state='COMPLETE', cells=list(CELLS), n=len(pop['ids']),
        chemical_groups=len(np.unique(pop['groups'])), supported=int(pop['support'].sum()),
        supported_chemical_groups=len(np.unique(pop['groups'][pop['support']])),
        metrics=metrics, comparisons=comparisons, by_cell=by_cell, mc_sensitivity=mc,
        sample_count=SAMPLES, scope='fixed four-cell development pilot; not full R3 replacement',
        no_new_calibration_selection=True, no_confirmation_opened=True)
    write_json(ROOT/'summary.json', result)
    text = ['# Biological support-mask pilot', '',
        f"{result['n']} conditions, {result['chemical_groups']} chemical groups; "
        f"{result['supported']} supported conditions in {result['supported_chemical_groups']} chemical groups.",
        '', '| Arm | Supported Gamma CRPS | Supported NLL | All Gamma CRPS | NULL / selected | Mean value |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, row in metrics.items():
        d = row['decision']
        text.append(f"| {arm} | {row['supported']['crps']:.8f} | {row['supported']['nll']:.6f} | "
            f"{row['all']['crps']:.8f} | {d['null']} / {d['selected']} | {d['mean_value']:.8f} |")
    text.extend(['', 'Paired chemical-group intervals, full observable scores, five coverage levels and three integration seeds are in summary.json.',
        'All new branches use 60 epochs and 100,000 joint samples; CORE and left branches are reused.',
        'The sample is one outer fold at four fixed doses. No activation rule or architecture is selected using these query results.'])
    (ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', default=','.join(map(str, CELLS)))
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    selected = tuple(map(int, args.cells.split(',')))
    if any(i not in CELLS for i in selected):
        raise ValueError('Only fixed pilot cells are permitted')
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    with threadpool_limits(1):
        try:
            if not args.summarize_only:
                scope = enriched_scope()
                for index in selected:
                    write_json(ROOT/'status.json', dict(state='RUNNING', cell=index,
                        elapsed_seconds=time.monotonic()-started))
                    run_cell(index, scope)
            if all((ROOT/f'fold_{i}/complete.json').exists() for i in CELLS):
                summarize()
                write_json(ROOT/'status.json', dict(state='COMPLETE', elapsed_seconds=time.monotonic()-started))
        except Exception:
            write_json(ROOT/'status.json', dict(state='FAILED', traceback=traceback.format_exc(),
                elapsed_seconds=time.monotonic()-started))
            raise


if __name__ == '__main__':
    main()
