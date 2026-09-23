"""RxRx3 R3: reuse R2 CORE; fit only the prespecified optional modules."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from opal2.biology_kernel_evaluation import write_json
from opal2.dual_branch_features import apply_increment, select_strength
from opal2.empirical_radial_experiment import score
from opal2.eu_core_experiment import extra_seed_moments, select
from opal2.gram_oof_ridge import transform_target
from opal2.rxrx3_r3_cache import load_scope, load_cell
from opal2.rxrx3_r3_biology import load_rxrx3_biology_metadata
from scripts.run_r3_eu_modules_20260918 import ARMS, BASE_ARMS, fit_modules, select_module
from scripts.run_r2_core_comparison_20260917 import read_npz

ROOT = PROJECT/'runs/r3_rxrx3_modules_20260920_v1'
REPORT = PROJECT/'reports/r3_rxrx3_modules_20260920_v1'
SEED = 20260920
CORE_SEED = 20260918
SAMPLES = 100000
OFFSETS = (100000, 200000)


def enriched_scope():
    scope = load_scope()
    biology = load_rxrx3_biology_metadata(scope['data'])
    scope['data'].update(biology['arrays'])
    scope['biology_metadata'] = biology['metadata']
    scope['biology_report'] = biology['report']
    return scope


def prepare_manifest(scope):
    ROOT.mkdir(parents=True, exist_ok=True)
    spec = dict(dataset='RxRx3 approved chemical DEV only', n=len(scope['data']['ids']),
        chemical_groups=len(np.unique(scope['data']['groups'])), cells=40, doses=8,
        arms=list(ARMS), seed=SEED, core_sampling_seed=CORE_SEED, samples=SAMPLES,
        additional_mc_seed_offsets=list(OFFSETS), source=str(scope['source']),
        new_mean_fitting=False, new_core_distribution_fitting=False,
        inner_reference_training='three group-held-out REF folds; new adapters only',
        calibration_representatives='lexicographic ID-min per chemical group',
        biology_prior='reported compound-gene IC50/EC50 value in (0,1000] nM; not HUVEC-validated targets',
        moa_available=False, reference_context_policy='rxrx3_protocol_range_v1',
        input_modality='approved well profiles', raw_images_used=False,
        ema_jepa_tested=False, cross_dose_acquisition_tested=False,
        query_endpoints_changed=False, confirmation_opened=False,
        primary_comparisons=['BIO_STRUCTURED versus GELU/BIO_GENERIC/BIO_RANDOM',
            'COND_REP versus GELU/PCA_REP/DIRECT_REP/DESCRIPTORS_HGB',
            'BOTH_STRUCTURED versus both single-module arms'],
        fitting_epochs=60, representation_internal_early_stopping=True,
        strength_grid=[0., .25, .5, 1.], frozen_policy_lambda=.2,
        inference='development; paired chemical-group blocks across all doses; layout sensitivity')
    path = ROOT/'run_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != spec:
            raise ValueError('Existing experiment has another specification')
    else:
        write_json(path, spec)
        shutil.copy2(REPORT/'PROTOCOL.md', ROOT/'PROTOCOL.md')
    write_json(ROOT/'biology_metadata_audit.json', scope['biology_report'])
    return spec


def run_cell(index, scope, *, prepare_only=False):
    start = time.monotonic()
    folder = ROOT/f'fold_{index}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        return json.loads((folder/'complete.json').read_text())
    def status(stage, **more):
        write_json(folder/'status.json', dict(state='RUNNING', cell=index, stage=stage,
            elapsed_seconds=time.monotonic()-start, **more))
    try:
        status('loading original dose-specific caches')
        cell = load_cell(f'fold_{index}', scope=scope)
        data, part, means, stats, arrays, state = (cell[k] for k in
            ('data', 'part', 'means', 'stats', 'oldarrays', 'state'))
        meta = dict(scope['biology_metadata'])
        meta['units'] = [scope['biology_metadata']['units'][int(i)] for i in cell['global_rows']]
        np.testing.assert_array_equal([u['id'] for u in meta['units']], data['ids'])
        r, c, q = (part[k] for k in ('REF_FIT', 'DIST_CAL', 'DEV_EVAL'))
        raw, actual, observed, difference = (cell[k] for k in ('raw','actual','observed','difference'))
        original, source = cell['original'], Path(cell['source'])
        write_json(folder/'source_reuse_audit.json', cell['audit'])
        status('training full optional modules; frozen CORE')
        begin = time.monotonic()
        predictions = fit_modules(data, meta, part, means, raw, stats, arrays, folder,
            SEED+100*index, distribution_state=state, calibration_group_representatives=True)
        training_seconds = time.monotonic()-begin
        logamp = np.log(np.linalg.norm(data['Y'][:,0], axis=1))
        scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
        rawmean = means*scale+center
        calibrations = {}
        status('CAL-only zero-inclusive strength selection')
        begin = time.monotonic()
        for arm in BASE_ARMS:
            path = folder/(arm+'_calibration.json')
            if path.exists():
                choice = json.loads(path.read_text())
            else:
                choice = select_strength(rawmean[c], scale, arrays['cal_scatter_u'],
                    predictions['cal_residual'], logamp[c], data['groups'][c],
                    state['radial_reference_bandwidth'], predictions[arm+'_cal'],
                    representative_ids=data['ids'][c])
                write_json(path, choice)
            calibrations[arm] = choice
        conditional = select_module(calibrations, data['groups'][c])
        write_json(folder/'conditional_selection.json', conditional)
        calibration_seconds = time.monotonic()-begin
        if prepare_only:
            write_json(folder/'status.json', dict(state='PREPARED', cell=index,
                elapsed_seconds=time.monotonic()-start, integration_pending=True))
            return dict(state='PREPARED', cell=index)
        target = transform_target(raw[q], stats)
        np.testing.assert_allclose(target, original['actual_u'], atol=1e-11, rtol=1e-11)
        np.testing.assert_array_equal(means[q], original['mean_u'])
        norm2 = np.square(np.asarray(data['Y'][q,0], dtype=np.float64)).mean(1)
        absolute = np.log1p(difference[q]*norm2[:,None])
        evaluated, increments, arm_cells = {}, {}, {}
        for arm in ARMS:
            begin = time.monotonic()
            path = folder/(arm+'.npz')
            if arm == 'CORE':
                increment = np.zeros((len(q),2))
            elif arm == 'CONDITIONAL':
                increment = increments[conditional['arm']].copy()
            else:
                base = arm.removesuffix('_CAL')
                strength = calibrations[base]['strength'] if arm.endswith('_CAL') else 1.
                increment = strength*predictions[base+'_query']
            changed = np.any(increment != 0, axis=1)
            equal = next((a for a, v in increments.items() if np.array_equal(v,increment)), None)
            increments[arm] = increment
            status('100k joint distribution evaluation', arm=arm)
            cached = path.exists()
            if cached:
                saved = read_npz(path)
                np.testing.assert_array_equal(saved['ids'],data['ids'][q])
                np.testing.assert_array_equal(saved['increment'],increment)
                out = {k:v for k,v in saved.items() if k not in ('ids','groups','actual')}
            elif arm == 'CORE':
                out = {k:v.copy() for k,v in original.items() if k not in ('ids','groups','actual')}
            elif equal is not None:
                out = {k:v.copy() for k,v in evaluated[equal].items()}
            else:
                scatter = apply_increment(rawmean[q],scale,arrays['query_scatter_u'],increment)
                np.testing.assert_array_equal(scatter[~changed],arrays['query_scatter_u'][~changed])
                out = score(means[q],scatter,target,stats,actual[q],observed[q],
                    absolute,norm2,CORE_SEED+100*index,law=state['law'],
                    weights=arrays['radial_weights'],samples=SAMPLES)
                for key,value in out.items():
                    if key in original:
                        np.testing.assert_array_equal(value[~changed],original[key][~changed])
                out.update(mean_u=means[q],actual_u=target,scatter_u=scatter)
            out.update(increment=increment,resource_support=predictions['query_support'])
            out['brier'] = np.square(out['p_null']-(actual[q]<=0))
            for lam in (.2,0.):
                out[f'selected_lambda_{lam:g}'] = select(data['ids'][q],out['predicted'],out['p_null'],lam)
            np.savez_compressed(path,ids=data['ids'][q],actual=actual[q],**out)
            evaluated[arm] = out
            for offset in OFFSETS:
                dest = folder/f'{arm}_mc{offset}.npz'
                if dest.exists():
                    continue
                if arm == 'CORE':
                    shutil.copy2(source/f'AMP_EMP_LOCAL_mc{offset}.npz',dest)
                elif equal is not None:
                    shutil.copy2(folder/f'{equal}_mc{offset}.npz',dest)
                else:
                    mc = extra_seed_moments(means[q],out['scatter_u'],stats,law=state['law'],
                        weights=arrays['radial_weights'],seed=CORE_SEED+100*index+offset)
                    base_mc = read_npz(source/f'AMP_EMP_LOCAL_mc{offset}.npz')
                    for key in mc:
                        np.testing.assert_array_equal(mc[key][~changed],base_mc[key][~changed])
                    np.savez_compressed(dest,ids=data['ids'][q],**mc)
            arm_cells[arm] = dict(changed=int(changed.sum()),reused_equal_arm=equal,
                reused_disk=cached,elapsed_seconds_this_invocation=time.monotonic()-begin,
                selected_null=int(((actual[q]<=0)&out['selected_lambda_0.2']).sum()),
                crps=float(out['crps'].mean()))
            write_json(folder/'arm_progress.json',arm_cells)
            print(f'cell={index} arm={arm} changed={changed.sum()} elapsed={time.monotonic()-start:.1f}',flush=True)
        info = cell['cell_info']
        record = dict(fold=index,cell=index,outer_fold=info['outer_fold'],dose_um=info['dose_uM'],
            n=len(q),query_ids=data['ids'][q],query_global_rows=cell['global_rows'][q],
            arms=arm_cells,conditional=conditional,
            calibrations={a:v['strength'] for a,v in calibrations.items()},
            costs=json.loads((source/'summary.json').read_text())['costs'],
            module_seconds_this_invocation=training_seconds,calibration_seconds_this_invocation=calibration_seconds,
            elapsed_seconds_this_invocation=time.monotonic()-start,
            historical_CORE_cost_not_zero=True,protected_measurements_read=False)
        write_json(folder/'complete.json',record)
        write_json(folder/'status.json',dict(state='COMPLETE',cell=index,elapsed_seconds=time.monotonic()-start))
        return record
    except Exception:
        write_json(folder/'status.json',dict(state='FAILED',cell=index,
            elapsed_seconds=time.monotonic()-start,traceback=traceback.format_exc()))
        raise


def supervise(workers=2):
    """Run disjoint full-configuration cells; no timing-dependent model selection."""
    start = time.monotonic()
    scope = enriched_scope()
    prepare_manifest(scope)
    del scope
    gc.collect()
    remaining = [i for i in range(40) if not (ROOT/f'fold_{i}/complete.json').exists()]
    processes = []
    for worker in range(workers):
        cells = remaining[worker::workers]
        if not cells:
            continue
        log = ROOT/f'worker_{worker}.log'
        with log.open('ab') as stream:
            command = [sys.executable,'-u',str(Path(__file__).resolve()),'--cells',','.join(map(str,cells))]
            process = subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT)
        processes.append((process,worker))
    while processes:
        alive, failures = [], []
        for process,worker in processes:
            code = process.poll()
            if code is None:
                alive.append((process,worker))
            elif code != 0:
                failures.append(dict(worker=worker,returncode=code))
        if failures:
            # Other workers can finish their independent cells; preserve all artifacts.
            write_json(ROOT/'worker_failures.json',dict(failures=failures))
        processes = alive
        complete = sum((ROOT/f'fold_{i}/complete.json').exists() for i in range(40))
        write_json(ROOT/'status.json',dict(state='RUNNING',stage='R3 full modules',
            completed_cells=complete,total_cells=40,workers=[dict(pid=p.pid,worker=w) for p,w in processes],
            elapsed_seconds=time.monotonic()-start))
        if processes:
            time.sleep(30)
    if not all((ROOT/f'fold_{i}/complete.json').exists() for i in range(40)):
        write_json(ROOT/'status.json',dict(state='FAILED',stage='worker failed; see cell status and logs',
            completed_cells=sum((ROOT/f'fold_{i}/complete.json').exists() for i in range(40))))
        raise RuntimeError('Not all R3 cells completed')
    write_json(ROOT/'status.json',dict(state='RUNNING',stage='aggregate',completed_cells=40))
    subprocess.run([sys.executable,str(PROJECT/'scripts/report_r3_rxrx3_modules_20260920.py')],
        cwd=PROJECT,check=True)
    write_json(ROOT/'status.json',dict(state='COMPLETE',completed_cells=40,
        elapsed_seconds=time.monotonic()-start,report=str(ROOT/'REPORT.md')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells',help='Comma-separated cell indices; workers only')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--workers',type=int,default=2,choices=(1,2))
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if args.cells is not None:
            scope = enriched_scope()
            prepare_manifest(scope)
            cells = [int(i) for i in args.cells.split(',')]
            if len(set(cells)) != len(cells) or any(i<0 or i>=40 for i in cells):
                raise ValueError('Distinct cell IDs 0..39 required')
            for cell in cells:
                run_cell(cell,scope,prepare_only=args.prepare_only)
                gc.collect()
        else:
            if args.prepare_only:
                raise ValueError('Specify cells for prepare-only')
            supervise(args.workers)
