"""Wait for existing primary L, run two post-fit MC seeds, then refresh R2.

The main experiment is never launched by this finisher. Both additional jobs
require all existing fitted L checkpoints and use one CPU thread each.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'runs/jump_matched_l_20260918_v1'
REPORT = PROJECT/'reports/jump_matched_l_20260918_v1'
ARM = 'L_TRAIN_MATCHED_FULL'
OFFSETS = (100000, 200000)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # Runners may be between truncation and close of a status write.
        return None


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(f'.{os.getpid()}.partial.json')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def seed_complete(root, offset):
    suffix = '' if offset == 0 else f'_mc{offset}'
    state = read_json(root/f'status_seed_{offset}.json')
    summary = read_json(root/('summary'+suffix+'.json'))
    return bool(state and state.get('state') == 'COMPLETE' and summary and summary.get('complete'))


def require_existing_fits(root):
    paths = [root/f'fold_{f}'/'complete_original_L.npz' for f in range(5)]
    if any(not path.is_file() for path in paths):
        raise RuntimeError('All five primary L fits must exist before sensitivity; no refitting is allowed')
    return [str(path) for path in paths]


def thread_environment():
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    return env


def launch_seed(project, root, offset):
    if offset not in OFFSETS:
        raise ValueError('Finisher can launch only the two additional MC seeds')
    require_existing_fits(root)
    command = [sys.executable, str(project/'scripts/run_jump_matched_l_20260918.py'),
               '--seed-offset', str(offset), '--reuse-fitted-only']
    log_path = root/f'background_mc{offset}.log'
    with log_path.open('ab') as log:
        process = subprocess.Popen(command, cwd=project, env=thread_environment(),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, dict(pid=process.pid, command=command, log=str(log_path),
        started_utc=now(), seed_offset=offset, cpu_threads=1,
        purpose='post-fit MC sensitivity', additional_training_runs=0, additional_label_wells=0)


def refresh(project, root, stage):
    before = time.monotonic()
    command = [sys.executable, str(project/'scripts/summarize_r2_four_datasets_20260918.py'), '--refresh']
    log_path = root/f'closure_refresh_{stage}.log'
    with log_path.open('ab') as log:
        completed = subprocess.run(command, cwd=project, env=thread_environment(),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=False)
    result = dict(stage=stage, utc=now(), returncode=completed.returncode,
        wall_seconds=time.monotonic()-before, command=command, log=str(log_path))
    atomic_json(root/f'closure_refresh_{stage}.json', result)
    return result


def policy_from_arrays(saved, lam):
    import numpy as np
    ids, folds, actual = saved['ids'], saved['fold'], saved['actual']
    selected = np.zeros(len(ids), bool)
    for fold in np.unique(folds):
        q = np.flatnonzero(folds == fold)
        budget = int(np.floor(.25*len(q)))
        order = np.lexsort((ids[q], -(saved['predicted'][q]-lam*saved['p_null'][q])))
        selected[q[order[:budget//2]]] = True
    key = f'selected_lambda_{lam:g}'
    np.testing.assert_array_equal(selected, saved[key])
    chosen_actual = actual[selected]
    null_count = int((chosen_actual <= 0).sum())
    expected = float(saved['p_null'][selected].sum())
    return dict(activated=int(selected.sum()), extra_wells=int(2*selected.sum()),
        null_selected=null_count, total_value=float(chosen_actual.sum()),
        selected_mean_value=float(chosen_actual.mean()),
        predicted_null_count=expected, null_count_gap=null_count-expected,
        selected_ids=ids[selected].tolist()), selected


def build_sensitivity(root, report):
    """Use only completed prediction arrays; never resample or refit."""
    import numpy as np
    offsets = (0, *OFFSETS)
    arrays, summaries = {}, {}
    for offset in offsets:
        if not seed_complete(root, offset):
            raise RuntimeError(f'Seed {offset} is not complete')
        suffix = '' if offset == 0 else f'_mc{offset}'
        with np.load(root/(ARM+suffix+'.npz'), allow_pickle=False) as z:
            arrays[offset] = {key: z[key].copy() for key in (
                'ids', 'fold', 'actual', 'predicted', 'p_null', 'selected_lambda_0.2', 'selected_lambda_0')}
        summaries[offset] = read_json(root/('summary'+suffix+'.json'))
        if summaries[offset]['samples'] != 100000:
            raise ValueError('Every seed must preserve 100,000 full-profile draws')
        if offset:
            for key in ('ids', 'fold', 'actual'):
                np.testing.assert_array_equal(arrays[offset][key], arrays[0][key])
    records, ranges = [], {}
    for offset in offsets:
        saved, policies = arrays[offset], {}
        for lam in (.2, 0.):
            key = f'lambda_{lam:g}'
            policy, selected = policy_from_arrays(saved, lam)
            original = arrays[0]['selected_'+key]
            shared, union = int((selected & original).sum()), int((selected | original).sum())
            policy.update(list_symmetric_difference=int((selected != original).sum()),
                list_symmetric_difference_from_primary=int((selected != original).sum()),
                intersection_with_primary=shared, jaccard_with_primary=shared/union,
                added_ids=saved['ids'][selected & ~original].tolist(),
                removed_ids=saved['ids'][original & ~selected].tolist())
            policies[key] = policy
        records.append(dict(seed_offset=offset, samples=100000, policies=policies,
            prediction_abs_difference_from_primary=dict(
                gamma_mean=float(np.abs(saved['predicted']-arrays[0]['predicted']).mean()),
                gamma_max=float(np.abs(saved['predicted']-arrays[0]['predicted']).max()),
                p_null_mean=float(np.abs(saved['p_null']-arrays[0]['p_null']).mean()),
                p_null_max=float(np.abs(saved['p_null']-arrays[0]['p_null']).max()))))
    for lam in (.2, 0.):
        key = f'lambda_{lam:g}'
        policies = [record['policies'][key] for record in records]
        ranges[key] = {field+'_min': min(p[field] for p in policies) for field in (
            'null_selected', 'total_value', 'predicted_null_count', 'list_symmetric_difference_from_primary')}
        ranges[key].update({field+'_max': max(p[field] for p in policies) for field in (
            'null_selected', 'total_value', 'predicted_null_count', 'list_symmetric_difference_from_primary')})
        ranges[key]['selected_null_count_min'] = ranges[key]['null_selected_min']
        ranges[key]['selected_null_count_max'] = ranges[key]['null_selected_max']
        stable = np.logical_and.reduce([arrays[offset]['selected_'+key] for offset in offsets])
        any_selected = np.logical_or.reduce([arrays[offset]['selected_'+key] for offset in offsets])
        ranges[key].update(all_seed_intersection=int(stable.sum()), all_seed_union=int(any_selected.sum()),
            selected_in_every_seed_ids=arrays[0]['ids'][stable].tolist())
    seconds = {str(offset): float(summaries[offset]['MC_and_scoring_wall_seconds']) for offset in offsets}
    payload = dict(complete=True, arm=ARM, n=len(arrays[0]['ids']), created_utc=now(),
        training_runs=5, additional_training_runs=0, additional_label_wells=0,
        primary_seed_offset=0, additional_seed_offsets=list(OFFSETS), samples_per_seed=100000,
        per_seed=records, policy_ranges=ranges,
        monte_carlo_sensitivity={ARM: records[1:]},
        compute=dict(seed_MC_and_scoring_wall_seconds=seconds,
            additional_MC_only_seconds=sum(seconds[str(offset)] for offset in OFFSETS),
            parallel_wall_time_is_not_sum=True, extra_compute_is_development_sensitivity=True,
            additional_deployment_cost=0, additional_training_time_seconds=0),
        scope='Three fixed 100k integration seeds of the same five fitted L models; no seed selected by outcome',
        original_endpoint_dimensions=3617, full_residual_sampling=True,
        final_opened=False, fifth_repeat_opened=False)
    atomic_json(root/'mc_sensitivity.json', payload)
    report.mkdir(parents=True, exist_ok=True)
    atomic_json(report/'mc_sensitivity.json', payload)
    lines = ['# JUMP L Monte Carlo sensitivity', '',
        'Three 100,000-draw integrations reuse the same five complete L fits. The additional seeds do not train models or consume assay labels.', '',
        '| Policy | NULL range | Value range | Max list symmetric difference from primary |',
        '|---|---:|---:|---:|']
    for key, row in ranges.items():
        lines.append(f"| {key} | {row['null_selected_min']}–{row['null_selected_max']} | {row['total_value_min']:.6f}–{row['total_value_max']:.6f} | {row['list_symmetric_difference_from_primary_max']} |")
    lines += ['', 'The primary seed remains the reported arm. Extra integration computation is development sensitivity and is excluded from deployment acquisition cost.']
    for folder in (root, report):
        (folder/'MC_SENSITIVITY.md').write_text('\n'.join(lines)+'\n')
    return payload


def run(project=PROJECT, root=ROOT, report=REPORT, poll_seconds=60):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'finalizer.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('Another L finisher is already running') from error
    progress_path = root/'finalizer_progress.json'
    progress = read_json(progress_path) or dict(children={}, refreshes={})
    started = time.monotonic()
    def status(state, **fields):
        atomic_json(progress_path, progress)
        record = dict(state=state, pid=os.getpid(), updated_utc=now(),
            elapsed_seconds=time.monotonic()-started, **fields)
        atomic_json(root/'finalizer_status.json', record)
        print(json.dumps(record), flush=True)
    try:
        while not seed_complete(root, 0):
            primary = read_json(root/'status_seed_0.json') or {}
            if primary.get('state') == 'FAILED':
                raise RuntimeError('Primary L failed; finisher will not repeat the primary run')
            if primary.get('pid') and not alive(primary['pid']):
                raise RuntimeError('Primary L process exited before completion; finisher will not repeat it')
            status('WAITING_PRIMARY', primary_pid=primary.get('pid'),
                completed_objects=primary.get('completed_objects'), primary_is_restarted=False)
            time.sleep(poll_seconds)
        require_existing_fits(root)
        if progress['refreshes'].get('primary_complete', {}).get('returncode') != 0:
            status('PRIMARY_COMPLETE_REFRESH')
            progress['refreshes']['primary_complete'] = refresh(project, root, 'primary_complete')
        owned = {}
        for offset in OFFSETS:
            if seed_complete(root, offset):
                continue
            state = read_json(root/f'status_seed_{offset}.json') or {}
            previous = progress['children'].get(str(offset), {})
            candidate_pid = state.get('pid') or previous.get('pid')
            if alive(candidate_pid):
                progress['children'][str(offset)] = dict(previous, pid=candidate_pid, adopted_existing_process=True)
            else:
                process, record = launch_seed(project, root, offset)
                owned[offset] = process
                progress['children'][str(offset)] = record
                atomic_json(progress_path, progress)
        while not all(seed_complete(root, offset) for offset in OFFSETS):
            children = {}
            for offset in OFFSETS:
                state = read_json(root/f'status_seed_{offset}.json') or {}
                if seed_complete(root, offset):
                    children[str(offset)] = dict(state='COMPLETE')
                    continue
                process = owned.get(offset)
                running = process.poll() is None if process is not None else alive(progress['children'][str(offset)]['pid'])
                if state.get('state') == 'FAILED' or not running:
                    raise RuntimeError(f'Additional MC seed {offset} failed or exited before completion')
                children[str(offset)] = dict(state='RUNNING', pid=progress['children'][str(offset)]['pid'],
                    completed_objects=state.get('completed_objects', 0))
            status('RUNNING_SENSITIVITY', children=children, cpu_threads_per_child=1,
                additional_training_runs=0, additional_label_wells=0)
            time.sleep(poll_seconds)
        status('AGGREGATING_MC_SENSITIVITY')
        payload = build_sensitivity(root, report)
        status('FINAL_REFRESH', additional_MC_only_seconds=payload['compute']['additional_MC_only_seconds'])
        progress['refreshes']['all_seeds_complete'] = refresh(project, root, 'all_seeds_complete')
        if progress['refreshes']['all_seeds_complete']['returncode']:
            raise RuntimeError('MC sensitivity complete, but four-dataset refresh failed; see refresh log')
        status('COMPLETE', sensitivity=str(root/'mc_sensitivity.json'),
            primary_repeated=False, additional_training_runs=0, additional_deployment_cost=0)
    except BaseException:
        status('FAILED', traceback=traceback.format_exc())
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('poll interval must be between 1 and 60 seconds')
    if args.aggregate_only:
        build_sensitivity(ROOT, REPORT)
    else:
        run(poll_seconds=args.poll_seconds)
