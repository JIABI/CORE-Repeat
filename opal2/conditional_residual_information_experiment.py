"""Opened-LINCS residual-information comparison around the frozen STATE50 CORE.

No query outcome selects a model, descriptor, epoch, scatter multiplier or rule.
The geometric projections are predictive-error directions, not physical noise.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .biology_borrowing_experiment import read_json, read_npz, scalar_metrics, paired_intervals
from .biology_kernel_evaluation import write_json
from .conditional_joint_error_experiment import observable_forward
from .conditional_residual_information import (BOOST_CONFIG, error_targets, ScalePredictor,
    fit_residual_state, bounded_scale_ratio, extend_scatter)
from .empirical_radial import fit_radial
from .frozen_acquisition_policy import select_frozen_cohort_plan
from .gram_geometry import profiles_to_gram, gram_to_coordinates
from .lincs_biology_experiment import load_data
from .radial_mixture_evaluation import evaluate_radial_mixtures
from .residual_descriptors import (ResidualDescriptorTransformer,
    load_normalized_plate_controls, information_availability)

PROJECT = Path(__file__).resolve().parents[1]
SEED = 20260916
SAMPLES = 100000
PLAN = 'protocols/historical/CONDITIONAL_RESIDUAL_INFORMATION_PLAN_20260916.md'
REMOVALS = {
    'NO_CELL_COUNT': ('cell_count',),
    'NO_TEXTURE': ('within_object_texture',),
    'NO_DISTANCE': ('train_distance',),
    'NO_RELIABILITY': ('reliability',),
    'NO_CONTROL_CONTEXT': ('plate_controls', 'position_qc'),
}
ARMS = ('CORE', 'DESCRIPTORS', 'DESCRIPTORS_LATENT', *REMOVALS)


def columns_without(blocks, excluded):
    return np.asarray(sorted({j for b, cols in blocks.items() if b not in excluded for j in cols}), int)


def calibrate_extended_scatter(mean_cal, mean_query, scale, scatter_cal, scatter_query,
                               residual_cal, ratio_cal, ratio_query):
    """Calibration sees calibration errors only; no query target argument exists."""
    c = extend_scatter(mean_cal, scale, scatter_cal, ratio_cal)
    q = extend_scatter(mean_query, scale, scatter_query, ratio_query)
    radii = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(c), residual_cal[..., None])[..., 0], axis=1)
    return c, q, radii, fit_radial(radii)


def train_fold(data, metadata, fit, controls, folder, fold):
    if (folder/'conditioning_complete.json').exists():
        saved = read_npz(folder/'conditioning_predictions.npz')
        np.testing.assert_array_equal(saved['ids'], data['ids'])
        np.testing.assert_array_equal(saved['fit_ids'], data['ids'][fit])
        return saved
    nested_file = folder/'nested_core/residuals.npz'
    nested_summary = folder/'nested_core/summary.json'
    while not nested_summary.exists():
        stage = folder.parent/'nested_stage_status.json'
        if stage.exists() and read_json(stage).get('state') == 'FAILED':
            raise RuntimeError('Nested residual stage failed; inspect its log')
        print(f'waiting for honest residuals fold={fold}', flush=True)
        time.sleep(20)
    nested = read_npz(nested_file)
    np.testing.assert_array_equal(nested['ids'], data['ids'][fit])
    np.testing.assert_array_equal(nested['prediction_count'], np.ones(len(fit), int))
    energies = error_targets(nested['raw_mean'], nested['raw_covariance'], nested['raw_residual'])
    transformer = ResidualDescriptorTransformer.fit(data, metadata, fit, plate_controls=controls, seed=SEED+fold)
    desc = transformer.transform(data, metadata)
    amp_columns = np.asarray(desc.blocks['amplitude']+desc.blocks['context'], int)
    amplitude_model = ScalePredictor.fit(desc.values[fit], energies, amp_columns, data['ids'][fit], seed=SEED+fold)
    amplitude = amplitude_model.predict(desc.values)
    full_columns = np.arange(desc.values.shape[1])
    models = {}; predictions = {}
    for arm in ('DESCRIPTORS', *REMOVALS):
        columns = full_columns if arm == 'DESCRIPTORS' else columns_without(desc.blocks, REMOVALS[arm])
        model = ScalePredictor.fit(desc.values[fit], energies, columns, data['ids'][fit], seed=SEED+fold)
        models[arm] = model
        predictions[arm] = model.predict(desc.values)
    state = fit_residual_state(desc.latent_input[fit], desc.values[fit], energies, amplitude[fit],
        data['ids'][fit], seed=SEED+100+fold,
        callback=lambda row: print(f'fold={fold} latent epoch={row["epoch"]} '
                                  f'train_nll={row["training_projection_nll"]:.6f}', flush=True))
    z = state.transform(desc.latent_input)
    augmented = np.column_stack((desc.values, z))
    models['DESCRIPTORS_LATENT'] = ScalePredictor.fit(augmented[fit], energies,
        np.arange(augmented.shape[1]), data['ids'][fit], seed=SEED+fold)
    predictions['DESCRIPTORS_LATENT'] = models['DESCRIPTORS_LATENT'].predict(augmented)
    ratios = {arm: bounded_scale_ratio(pred, amplitude) for arm, pred in predictions.items()}
    state.save(folder/'residual_state.pt')
    joblib.dump(dict(transformer=transformer, amplitude=amplitude_model, conditional=models), folder/'conditioners.joblib')
    write_json(folder/'descriptor_report.json', desc.report)
    write_json(folder/'latent_report.json', state.report)
    np.savez_compressed(folder/'conditioning_predictions.npz', ids=data['ids'], fit_ids=data['ids'][fit],
        fit_energies=energies, amplitude_scale=amplitude, latent=z, **ratios)
    write_json(folder/'conditioning_complete.json', dict(fold=fold, fit_n=len(fit),
        booster=BOOST_CONFIG, arms=list(ratios), input_features=desc.names,
        excluded_blocks=REMOVALS, no_query_supervision=True,
        ratio_interpretation='Conditional/amplitude error-scale ratio, transferred from honest inner recipe to outer CORE scatter',
        means_changed=False, latent=state.report))
    return read_npz(folder/'conditioning_predictions.npz')


def summarize(stores, data, metadata, cells, source, output):
    ids, groups = data['ids'], data['groups']
    actual = stores['CORE']['actual']; layouts = np.asarray([u['layout_block'] for u in metadata['units']])
    all_rows = np.ones(len(ids), bool)
    metrics = {}; comparisons = {}
    for arm, out in stores.items():
        out['brier'] = (out['p_null']-(actual <= 0))**2
        out['policy_value'] = out['selected']*actual
        out['policy_null'] = out['selected']*(actual <= 0)
        chosen = out['selected'].astype(bool)
        metrics[arm] = dict(full=scalar_metrics(out, actual, all_rows),
            selected_n=int(chosen.sum()), selected_null=int((actual[chosen] <= 0).sum()),
            selected_mean=float(actual[chosen].mean()),
            changed_selected_membership=int(np.count_nonzero(chosen != stores['CORE']['selected'])),
            by_fold=[dict(fold=f, metrics=scalar_metrics(out, actual, out['fold']==f)) for f in range(5)])
        np.savez_compressed(output/(arm+'.npz'), ids=ids, groups=groups, layout=layouts, **out)
    pairs = [('DESCRIPTORS', 'CORE'), ('DESCRIPTORS_LATENT', 'DESCRIPTORS')]
    pairs += [(a, 'DESCRIPTORS') for a in REMOVALS]
    for a,b in pairs:
        comparisons[a+'_minus_'+b] = dict(
            distribution=paired_intervals(stores[a], stores[b], all_rows, groups, layouts,
                keys=('crps','brier','nll','energy','single_crps','pair_crps','average_crps','absolute_pair_crps')),
            allocation=paired_intervals(stores[a], stores[b], all_rows, groups, layouts,
                keys=('policy_value','policy_null')))
    lookup = {i:j for j,i in enumerate(ids)}
    lambda0 = np.zeros(len(ids), bool); rand_gain = 0.; rand_null = 0.
    for cell in cells:
        q = np.asarray([lookup[i] for i in cell['query_ids']]); k = cell['budget']
        order = np.lexsort((ids[q], -stores['CORE']['predicted'][q]))
        lambda0[q[order[:k]]] = True
        rand_gain += k*actual[q].mean(); rand_null += k*(actual[q] <= 0).mean()
    summary = dict(state='COMPLETE', source=str(source), n=len(ids), metrics=metrics, comparisons=comparisons,
        samples=SAMPLES, means_changed=False, endpoint_changed=False, main_rule_changed=False,
        formal_certificate=False, query_used_for_fitting=False,
        controls=dict(core_lambda0=dict(selected_n=int(lambda0.sum()), selected_null=int((actual[lambda0] <= 0).sum()),
            selected_mean=float(actual[lambda0].mean())),
            uniform_random_expectation=dict(selected_mean=float(rand_gain/lambda0.sum()),selected_null=float(rand_null))),
        uncertainty='Paired chemical-group and layout bootstrap conditional on fixed development folds; not a new independent validation',
        scope='Two geometric conditional error scales and empirical radius; not identified shared/independent physical noise; not an arbitrary angular distribution')
    write_json(output/'summary.json', summary)
    lines = ['# Conditional residual information: opened LINCS development comparison', '',
        'All query means, outcomes, folds and budget decisions are unchanged in definition. '
        'The conditional distribution alone is modified. These are development comparisons, not certification.', '',
        '| Arm | Gamma CRPS | NULL Brier | NLL | Energy | Selected NULL / n | Selected value |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,m in metrics.items():
        s=m['full']['scores']
        lines.append(f'| {arm} | {s["crps"]:.6f} | {s["brier"]:.6f} | {s["nll"]:.6f} | '
                     f'{s["energy"]:.6f} | {m["selected_null"]}/{m["selected_n"]} | {m["selected_mean"]:.6f} |')
    lines += ['', 'Paired intervals, all five coverage levels, fold results and block-removal comparisons are in `summary.json`.',
        'CellProfiler texture summaries are not cell-to-cell heterogeneity. Normalized DMSO summaries are not raw plate noise variance.',
        'No matched potency annotations were introduced. The latent arm is a supervised four-dimensional residual conditioner, not teacher-based JEPA.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def run(source, output):
    start = time.monotonic(); source=Path(source).resolve(); root=Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    radial_source = Path(read_json(source/'summary.json')['source'])
    old = read_json(radial_source/'summary.json')
    manifest = read_json(Path(old['reference_run'])/'run_manifest.json')
    data, metadata = load_data(old['data_directory'])
    ids,groups=data['ids'],data['groups']; n=len(ids)
    if n != 1188 or ids.tolist() != manifest['ids']:
        raise ValueError('Opened development population differs')
    spec = dict(source=str(source), samples=SAMPLES, seed=SEED, arms=list(ARMS))
    if (root/'run_spec.json').exists():
        if read_json(root/'run_spec.json') != spec:
            raise ValueError('Run specification changed')
    else:
        write_json(root/'run_spec.json', spec)
    if not (root/'PROTOCOL.md').exists():
        shutil.copy2(PROJECT/PLAN, root/'PROTOCOL.md')
    if (root/'PROTOCOL.md').read_bytes() != (PROJECT/PLAN).read_bytes():
        raise ValueError('Protocol changed after fitting started')
    for file in (Path(__file__), PROJECT/'opal2/conditional_residual_information.py', PROJECT/'opal2/residual_descriptors.py'):
        destination=root/file.name
        if destination.exists() and destination.read_bytes() != file.read_bytes():
            raise ValueError('Implementation changed; use a separately declared run')
        shutil.copy2(file,destination)
    core=read_npz(source/'CORE.npz'); prior=read_npz(radial_source/'AMP_EMP_LOCAL.npz')
    np.testing.assert_array_equal(core['ids'],ids); np.testing.assert_array_equal(prior['ids'],ids)
    raw = gram_to_coordinates(profiles_to_gram(torch.tensor(data['Y']))).numpy()
    actual,observed,difference,_=observable_forward(raw)
    np.testing.assert_allclose(actual,core['actual'],rtol=1e-12,atol=1e-12)
    norm2=np.square(data['Y'][:,0]).mean(1); absolute=np.log1p(difference*norm2[:,None])
    controls=load_normalized_plate_controls(metadata,data['feature_names'],
        PROJECT/'reports/lincs_biology_preflight_20260915/metadata_audit/profile_cache')
    write_json(root/'information_availability.json',information_availability(data,metadata,controls))
    records={r['fold']:r for r in manifest['folds']}; lookup={i:j for j,i in enumerate(ids)}
    stores={a:{} for a in ARMS}; seen=np.zeros(n,int); trained={}
    for number,cell in enumerate(old['cells']):
        fold,half=cell['fold'],cell['half']; folder=root/f'cell_{fold}_{half}'; folder.mkdir(exist_ok=True)
        q=np.asarray([lookup[i] for i in cell['query_ids']]); cal=np.asarray([lookup[i] for i in cell['representative_ids']])
        fit=np.asarray(records[fold]['fit'],int)
        if set(groups[fit])&set(groups[np.r_[q,cal]]) or set(groups[q])&set(groups[cal]):
            raise ValueError('Cross-role chemical-group leakage')
        if fold not in trained:
            trained[fold]=train_fold(data,metadata,fit,controls,root/f'fold_{fold}',fold)
        ratios=trained[fold]; stats=read_json(Path(old['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json')
        scale,center=np.asarray(stats['u_scale']),np.asarray(stats['u_center'])
        raw_mean=prior['mean_u']*scale+center
        ref=read_npz(radial_source/f'cell_{fold}_{half}_radial.npz')
        np.testing.assert_array_equal(ref['cal_ids'],ids[cal]); np.testing.assert_array_equal(ref['query_ids'],ids[q])
        np.testing.assert_allclose(ref['cal_residual'],prior['actual_u'][cal]-prior['mean_u'][cal],rtol=1e-11,atol=1e-11)
        for arm in ARMS:
            path=folder/(arm+'.npz')
            if not path.exists():
                write_json(root/'status.json',dict(state='RUNNING',stage='distribution_evaluation',cell=number+1,
                    cells=10,arm=arm,elapsed_seconds=time.monotonic()-start))
                if arm=='CORE':
                    out={k:v[q].copy() for k,v in core.items() if v.shape[:1]==(n,) and k not in ('ids','groups','layout')}
                else:
                    c,s,radii,law=calibrate_extended_scatter(raw_mean[cal],raw_mean[q],scale,
                        ref['cal_amp_scatter'],prior['scatter_u'][q],ref['cal_residual'],ratios[arm][cal],ratios[arm][q])
                    out=evaluate_radial_mixtures(prior['mean_u'][q],s,prior['actual_u'][q],stats,
                        actual[q],observed[q],absolute[q],norm2[q],cell['normal_seed'],law=law,
                        endpoint_weights={'CORE':ref['local_weights']},coefficients={arm:np.ones(1)},
                        samples=SAMPLES)[arm]
                    selection=select_frozen_cohort_plan(ids[q],out['predicted'],out['p_null'],cell['budget'])
                    out['selected']=np.asarray(selection.selected_mask,int)
                    out['mean_u']=prior['mean_u'][q].copy(); out['actual_u']=prior['actual_u'][q].copy()
                    out['actual']=actual[q]; out['fold']=np.full(len(q),fold,int)
                    np.savez_compressed(folder/(arm+'_calibration.npz'),cal_ids=ids[cal],query_ids=ids[q],
                        cal_scatter=c,query_scatter=s,radii=radii,ratio_cal=ratios[arm][cal],ratio_query=ratios[arm][q])
                    write_json(folder/(arm+'_law.json'),law)
                np.savez_compressed(path,ids=ids[q],**out)
                print(f'cell={number+1}/10 arm={arm} elapsed={time.monotonic()-start:.1f}s',flush=True)
            out=read_npz(path); np.testing.assert_array_equal(out.pop('ids'),ids[q])
            np.testing.assert_array_equal(out['mean_u'],prior['mean_u'][q])
            for key,value in out.items():
                if key not in stores[arm]: stores[arm][key]=np.empty((n,*value.shape[1:]),dtype=value.dtype)
                stores[arm][key][q]=value
        seen[q]+=1
    np.testing.assert_array_equal(seen,np.ones(n,int))
    for key in ('mean_u','predicted','p_null','crps','nll','selected'):
        np.testing.assert_array_equal(stores['CORE'][key],core[key])
    summarize(stores,data,metadata,old['cells'],source,root)
    write_json(root/'status.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-start))
    print('COMPLETE',str(root),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default=str(PROJECT/'runs/module_switches_20260916_v1/lincs_v2'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run(args.source,args.output)


if __name__=='__main__':
    main()
