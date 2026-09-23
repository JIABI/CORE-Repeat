"""Matched random-reference attribution around an unchanged biological mixture.

Reference weights are permuted only within legal amplitude strata, preserving
per-query positive weight multisets. Outcomes are never used for matching.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .biology_borrowing_experiment import read_json, read_npz, scalar_metrics, paired_intervals
from .biology_kernel_evaluation import write_json
from .conditional_joint_error_experiment import observable_forward
from .empirical_radial import fit_radial
from .frozen_acquisition_policy import select_frozen_cohort_plan
from .gram_geometry import profiles_to_gram, gram_to_coordinates
from .lincs_biology_experiment import load_data
from .module_switch_experiment import context_mask, biology_similarity
from .optional_radial_modules import supported_retrieval
from .radial_mixture_evaluation import evaluate_radial_mixtures

PROJECT = Path(__file__).resolve().parents[1]
SEED = 2026091607
REPLICATES = 20
SAMPLES = 100000
ALPHAS = ((.25, '025'), (.5, '050'), (1., '100'))
PLAN = 'protocols/historical/RANDOM_BIOLOGY_REFERENCE_PLAN_20260916.md'


def amplitude_bins(fit_amplitude, reference_amplitude):
    """Quintile boundaries use MODEL_FIT only; coincident edges stay valid."""
    fit, ref = np.asarray(fit_amplitude, float), np.asarray(reference_amplitude, float)
    if fit.ndim != 1 or ref.ndim != 1 or len(fit) < 5 or not np.isfinite(fit).all() or not np.isfinite(ref).all():
        raise ValueError('Finite aligned one-dimensional amplitudes are required')
    edges = np.quantile(fit, [.2, .4, .6, .8])
    return np.searchsorted(edges, ref, side='right'), edges


def matched_random_weights(weights, legal, donor_bins, donor_amplitude, *, seed):
    """Permute legal donor positions, independently per query and amplitude bin.

    Zero rows remain zero. The donor-map entry at an originally positive
    position names its randomized destination. Unsupported entries remain -1.
    """
    w, allowed = np.asarray(weights, float), np.asarray(legal)
    bins, amp = np.asarray(donor_bins), np.asarray(donor_amplitude, float)
    if (w.ndim != 2 or allowed.dtype != bool or allowed.shape != w.shape
            or bins.shape != (w.shape[1],) or amp.shape != bins.shape
            or not np.isfinite(w).all() or np.any(w < 0) or not np.isfinite(amp).all()
            or np.any((w > 0) & ~allowed)):
        raise ValueError('Original donor weights must be finite and legal')
    supported = w.sum(1) > 0
    if not np.allclose(w[supported].sum(1), 1., rtol=0, atol=1e-12):
        raise ValueError('Supported weight rows must be normalized')
    rng = np.random.default_rng(seed)
    random = np.zeros_like(w)
    mapping = np.full(w.shape, -1, dtype=int)
    fixed_mass = np.zeros(len(w))
    displacement = np.zeros(len(w))
    retained = np.zeros(len(w))
    for i in np.flatnonzero(supported):
        for b in np.unique(bins):
            pool = np.flatnonzero(allowed[i] & (bins == b))
            if not len(pool):
                continue
            dest = rng.permutation(pool)
            positive = w[i, pool] > 0
            original, selected = pool[positive], dest[positive]
            mapping[i, original] = selected
            random[i, selected] = w[i, original]
            if len(pool) == 1:
                fixed_mass[i] += w[i, original].sum()
            displacement[i] += np.sum(w[i, original] * np.abs(amp[selected]-amp[original]))
            retained[i] += np.sum(w[i, original] * (selected == original))
    np.testing.assert_array_equal(np.sort(random, axis=1), np.sort(w, axis=1))
    np.testing.assert_array_equal(random[~allowed], np.zeros(np.count_nonzero(~allowed)))
    for b in np.unique(bins):
        np.testing.assert_allclose(random[:, bins == b].sum(1), w[:, bins == b].sum(1), rtol=0, atol=1e-14)
    return random, dict(donor_mapping=mapping, fixed_mass=fixed_mass,
        weighted_log_amplitude_displacement=displacement,
        retained_weight_mass=retained, changed_weight_rows=np.any(random != w, axis=1),
        positive_count=(random > 0).sum(1), legal_count=allowed.sum(1),
        ess=np.divide(1., np.square(random).sum(1), out=np.zeros(len(w)), where=supported))


def coefficient_plans(target, moa):
    """Three prespecified intensities; no gate fitting or outcome selection."""
    n = len(target)
    plans = {'CORE': np.tile([1., 0., 0.], (n, 1))}
    for j, (name, support) in enumerate((('TARGET', target), ('MOA', moa)), 1):
        for alpha, label in ALPHAS:
            c = np.zeros((n, 3)); c[:, 0] = 1-alpha*support; c[:, j] = alpha*support
            plans[name+'_A'+label] = c
    return plans


def arm_metrics(out, actual, support, core):
    selected = out['selected'].astype(bool)
    return dict(supported=scalar_metrics(out, actual, support),
        full=scalar_metrics(out, actual, np.ones(len(actual), bool)),
        selected_n=int(selected.sum()), selected_null=int((actual[selected] <= 0).sum()),
        selected_mean=float(actual[selected].mean()),
        changed_selected_membership=int(np.count_nonzero(selected != core['selected'])))


def aggregate(root, source, data, metadata, support, replicates):
    """Average scores, not distributions or selected lists, across randomizations."""
    ids, groups = data['ids'], data['groups']
    layout = np.array([u['layout_block'] for u in metadata['units']])
    core = read_npz(source/'CORE.npz'); actual = core['actual']
    result = {}
    for relation in ('TARGET', 'MOA'):
        mask = support[relation]
        for _, label in ALPHAS:
            arm = relation+'_A'+label
            real = read_npz(source/(arm+'.npz'))
            draws = [read_npz(root/f'replicate_{r:02d}'/(arm+'.npz')) for r in range(replicates)]
            keys = ('crps', 'brier', 'nll', 'energy', 'single_crps', 'pair_crps',
                    'average_crps', 'absolute_pair_crps', 'joint_coverage_by_level')
            expected = {k: np.mean([d[k] for d in draws], axis=0) for k in keys}
            expected['policy_value'] = np.mean([d['selected']*actual for d in draws], axis=0)
            expected['policy_null'] = np.mean([d['selected']*(actual <= 0) for d in draws], axis=0)
            real['policy_value'] = real['selected']*actual
            real['policy_null'] = real['selected']*(actual <= 0)
            result[arm] = dict(real_biology=arm_metrics(real, actual, mask, core),
                random_replicates=[arm_metrics(d, actual, mask, core) for d in draws],
                real_minus_expected_random_supported=paired_intervals(real, expected, mask, groups, layout),
                expected_random_minus_core_supported=paired_intervals(expected, core, mask, groups, layout),
                real_minus_expected_random_full=paired_intervals(real, expected, np.ones(len(ids), bool), groups, layout),
                allocation_real_minus_expected_random=paired_intervals(real, expected,
                    np.ones(len(ids), bool), groups, layout, keys=('policy_value', 'policy_null')),
                expected_random_supported_scores={k:float(expected[k][mask].mean()) for k in keys if k != 'joint_coverage_by_level'},
                expected_random_full_joint_coverage=expected['joint_coverage_by_level'].mean(0))
    return result


def run(source, output, *, resume=False):
    start = time.monotonic(); source, root = Path(source).resolve(), Path(output).resolve()
    original = read_json(source/'summary.json')
    module_source = Path(original['source'])
    radial_source = Path(read_json(module_source/'summary.json')['source'])
    radial = read_json(radial_source/'summary.json')
    manifest = read_json(Path(radial['reference_run'])/'run_manifest.json')
    data, metadata = load_data(radial['data_directory'])
    ids, groups = data['ids'], data['groups']; n = len(ids)
    if n != 1188 or ids.tolist() != manifest['ids']:
        raise ValueError('Opened development population changed')
    if root.exists() and not resume:
        raise FileExistsError(root)
    if not root.exists():
        root.mkdir(parents=True)
        shutil.copy2(PROJECT/PLAN, root/'PROTOCOL.md')
        shutil.copy2(Path(__file__), root/Path(__file__).name)
        write_json(root/'run_spec.json', dict(source=str(source), samples=SAMPLES,
            replicates=REPLICATES, seed=SEED, started_unix=time.time()))
    spec = read_json(root/'run_spec.json')
    if any(spec[k] != v for k,v in dict(source=str(source),samples=SAMPLES,replicates=REPLICATES,seed=SEED).items()):
        raise ValueError('Resume specification differs')
    if (root/'PROTOCOL.md').read_bytes() != (PROJECT/PLAN).read_bytes():
        raise ValueError('Protocol changed after run began')
    if (root/Path(__file__).name).read_bytes() != Path(__file__).read_bytes():
        raise ValueError('Implementation changed; start a separately declared run')
    core = read_npz(source/'CORE.npz'); prior = read_npz(radial_source/'AMP_EMP_LOCAL.npz')
    np.testing.assert_array_equal(ids, core['ids']); np.testing.assert_array_equal(ids, prior['ids'])
    raw = gram_to_coordinates(profiles_to_gram(torch.tensor(data['Y']))).numpy()
    actual, observed, differences, _ = observable_forward(raw)
    np.testing.assert_allclose(actual, core['actual'], rtol=1e-12, atol=1e-12)
    amp = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    norm2 = np.square(data['Y'][:, 0]).mean(1)
    absolute = np.log1p(differences*norm2[:, None])
    lookup = {v:i for i,v in enumerate(ids)}
    folds = {r['fold']:r for r in manifest['folds']}
    all_support = {r:np.zeros(n, bool) for r in ('TARGET','MOA')}
    prepared = []
    for cell_number, cell in enumerate(radial['cells']):
        fold, half = cell['fold'], cell['half']
        q = np.array([lookup[v] for v in cell['query_ids']])
        cal = np.array([lookup[v] for v in cell['representative_ids']])
        fit = np.asarray(folds[fold]['fit'], int)
        if set(groups[fit]) & set(groups[np.r_[q,cal]]) or set(groups[q]) & set(groups[cal]):
            raise ValueError('Group leakage across fixed roles')
        saved = read_npz(source/f'cell_{fold}_{half}'/'reference_plans.npz')
        ref = read_npz(radial_source/f'cell_{fold}_{half}_radial.npz')
        np.testing.assert_array_equal(saved['query_ids'], ids[q]); np.testing.assert_array_equal(saved['reference_ids'], ids[cal])
        np.testing.assert_array_equal(saved['endpoint_CORE'], ref['local_weights'])
        sims = biology_similarity(data, metadata, q, cal)
        channels = [supported_retrieval(s) for s in sims]
        legal = context_mask(metadata, q, cal) & (groups[q, None] != groups[None, cal])
        bins, edges = amplitude_bins(amp[fit], amp[cal])
        for relation, channel in zip(('TARGET','MOA'), channels):
            all_support[relation][q] = channel['supported']
            np.testing.assert_array_equal(channel['weights'][channel['supported']], saved['endpoint_'+relation][channel['supported']])
        prepared.append(dict(cell=cell, q=q, cal=cal, fit=fit, ref=ref, channels=channels,
            legal=legal, bins=bins, edges=edges,
            stats=read_json(Path(radial['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json')))
    write_json(root/'preflight.json', dict(n=n,cells=len(prepared),replicates=REPLICATES,
        supported={k:int(v.sum()) for k,v in all_support.items()},
        samples=SAMPLES,exact_original_endpoints=True,roles_disjoint=True))
    for rep in range(REPLICATES):
        repdir = root/f'replicate_{rep:02d}'; repdir.mkdir(exist_ok=True)
        if (repdir/'summary.json').exists():
            continue
        stores = {}; diagnostics = []; seen = np.zeros(n,int)
        for number, prepared_cell in enumerate(prepared):
            cell = prepared_cell['cell']; q, cal = prepared_cell['q'], prepared_cell['cal']
            folder = repdir/f"cell_{cell['fold']}_{cell['half']}"; folder.mkdir(exist_ok=True)
            arms = [r+'_A'+l for r in ('TARGET','MOA') for _,l in ALPHAS]
            if not (folder/'complete.json').exists():
                endpoints = {'CORE':prepared_cell['ref']['local_weights']}
                maps = {}; cell_diagnostic = {}
                for j,(relation,channel) in enumerate(zip(('TARGET','MOA'),prepared_cell['channels'])):
                    allowed = prepared_cell['legal'] & data[relation.lower()+'_mask'][cal][None].astype(bool)
                    w, diag = matched_random_weights(channel['weights'], allowed,
                        prepared_cell['bins'], amp[cal], seed=SEED+100000*rep+100*number+j)
                    w[~channel['supported']] = endpoints['CORE'][~channel['supported']]
                    endpoints[relation] = w
                    maps[relation+'_weights'] = w; maps[relation+'_donor_mapping'] = diag.pop('donor_mapping')
                    cell_diagnostic[relation] = diag
                support = [c['supported'] for c in prepared_cell['channels']]
                plans = coefficient_plans(*support)
                write_json(root/'status.json',dict(state='RUNNING',replicate=rep+1,replicates=REPLICATES,
                    cell=number+1,cells=len(prepared),elapsed_seconds=time.monotonic()-start))
                scores = evaluate_radial_mixtures(prior['mean_u'][q], prior['scatter_u'][q],
                    prior['actual_u'][q], prepared_cell['stats'], actual[q], observed[q],absolute[q],norm2[q],
                    cell['normal_seed'],law=fit_radial(prepared_cell['ref']['amplitude_radii']),
                    endpoint_weights=endpoints,coefficients=plans,samples=SAMPLES,
                    baseline_saved={k:v[q] for k,v in core.items() if isinstance(v,np.ndarray) and v.shape[:1]==(n,)})
                for arm,out in scores.items():
                    if arm == 'CORE':
                        for key in ('predicted','p_null','crps','nll'):
                            np.testing.assert_array_equal(out[key],core[key][q])
                        continue
                    selected = select_frozen_cohort_plan(ids[q],out['predicted'],out['p_null'],cell['budget'])
                    out['selected'] = np.asarray(selected.selected_mask,int)
                    np.savez_compressed(folder/(arm+'.npz'),ids=ids[q],actual=actual[q],**out)
                    eligible = support[0 if arm.startswith('TARGET') else 1]
                    for key in ('predicted','p_null','crps','nll','brier','joint_coverage_by_level'):
                        np.testing.assert_array_equal(out[key][~eligible],core[key][q][~eligible])
                np.savez_compressed(folder/'reference_maps.npz',query_ids=ids[q],donor_ids=ids[cal],
                    donor_bins=prepared_cell['bins'],amplitude_edges=prepared_cell['edges'],**maps)
                write_json(folder/'complete.json',dict(cell=number,seed_base=SEED+100000*rep+100*number,
                    diagnostic=cell_diagnostic,query_ids=ids[q].tolist(),budget=cell['budget']))
            diagnostics.append(read_json(folder/'complete.json'))
            for arm in arms:
                out = read_npz(folder/(arm+'.npz'))
                np.testing.assert_array_equal(out.pop('ids'),ids[q]); np.testing.assert_array_equal(out.pop('actual'),actual[q])
                stores.setdefault(arm,{})
                for key,value in out.items():
                    if key not in stores[arm]: stores[arm][key] = np.empty((n,*value.shape[1:]),dtype=value.dtype)
                    stores[arm][key][q] = value
            seen[q] += 1
            print(f'randomization={rep+1}/{REPLICATES} cell={number+1}/10 elapsed={time.monotonic()-start:.1f}s',flush=True)
        np.testing.assert_array_equal(seen,np.ones(n,int))
        for arm,out in stores.items():
            np.savez_compressed(repdir/(arm+'.npz'),ids=ids,actual=actual,**out)
        metrics = {arm:arm_metrics(out,actual,all_support[arm.split('_')[0]],core) for arm,out in stores.items()}
        write_json(repdir/'summary.json',dict(state='COMPLETE',replicate=rep,metrics=metrics,
            diagnostics=diagnostics,elapsed_session_seconds=time.monotonic()-start))
    results = aggregate(root,source,data,metadata,all_support,REPLICATES)
    write_json(root/'summary.json',dict(state='COMPLETE',source=str(source),n=n,replicates=REPLICATES,
        samples=SAMPLES,results=results,world_model_retrained=False,formal_certificate=False,
        randomization_pvalue_claimed=False,selected_randomization=False,endpoint_changed=False,
        uncertainty='Paired chemistry/layout bootstrap conditional on fixed CORE/reference banks; randomization spread reported separately',
        elapsed_session_seconds=time.monotonic()-start))
    write_json(root/'status.json',dict(state='COMPLETE',replicates_complete=REPLICATES,
        elapsed_session_seconds=time.monotonic()-start))
    print('COMPLETE',str(root),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',default=str(PROJECT/'runs/biology_borrowing_diagnostic_20260916_v2'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        run(args.source,args.output,resume=args.resume)


if __name__ == '__main__':
    main()
