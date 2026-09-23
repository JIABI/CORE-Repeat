"""Run all five honest MODEL_FIT residual stages without changing frozen CORE."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False)+'\n')


def prepare(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = PROJECT/'protocols/historical/CONDITIONAL_RESIDUAL_INFORMATION_PLAN_20260916.md'
    destination = root/'PROTOCOL.md'
    if destination.exists():
        if destination.read_bytes() != protocol.read_bytes():
            raise ValueError('Run already contains a different protocol')
    else:
        shutil.copy2(protocol, destination)
    if (root/'nested_stage_process.json').exists():
        raise FileExistsError('Nested stage has already been launched; inspect its existing state')
    return root


def execute(root):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from opal2.lincs_biology_experiment import load_data
    from opal2.nested_core_residuals import fit_nested_core_residuals, _take
    source = PROJECT/'runs/lincs_state_biology_20260916_v1'
    manifest = json.loads((source/'run_manifest.json').read_text())
    data, metadata = load_data(manifest['data_directory'])
    if data['ids'].tolist() != manifest['ids'] or data['groups'].tolist() != manifest['groups']:
        raise ValueError('Original opened MODEL_FIT identities changed')
    torch.set_num_threads(1)
    started, completed = time.monotonic(), []
    status = dict(stage='nested_complete_core_residuals', state='RUNNING', started_utc=now(),
        source_run=str(source), threads=1, inner_folds=3, completed_folds=completed,
        main_core_changed=False, protected_data_opened=False)
    save(root/'nested_stage_status.json', status)
    try:
        with threadpool_limits(limits=1):
            for record in manifest['folds']:
                fold = int(record['fold'])
                fit = np.asarray(record['fit'], int)
                folder = root/f'fold_{fold}'/'nested_core'
                if folder.exists():
                    raise FileExistsError('Existing partial/completed nested fit is preserved: '+str(folder))
                before = time.monotonic()
                print(json.dumps(dict(event='FOLD_STARTED', fold=fold, n_fit=len(fit), utc=now())), flush=True)
                result = fit_nested_core_residuals(_take(data, fit), metadata, folder,
                                                   seed=20260916+1000*fold, n_splits=3)
                completion = dict(fold=fold, n_fit=len(fit), seconds=time.monotonic()-before,
                    output=str(folder), raw_geometry_mse=result['summary']['raw_geometry_mse'])
                completed.append(completion)
                status.update(current_fold=fold, completed_folds=completed, elapsed_seconds=time.monotonic()-started)
                save(root/'nested_stage_status.json', status)
                print(json.dumps(dict(event='FOLD_COMPLETE', **completion)), flush=True)
        status.update(state='COMPLETE', completed_utc=now(), elapsed_seconds=time.monotonic()-started)
        save(root/'nested_stage_status.json', status)
    except BaseException as error:
        status.update(state='FAILED', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.monotonic()-started, failed_utc=now())
        save(root/'nested_stage_status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.execute:
        execute(root)
        return
    root = prepare(root)
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execute', '--output', str(root)]
    with (root/'nested_stage_stdout.log').open('xb') as out, (root/'nested_stage_stderr.log').open('xb') as err:
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=out, stderr=err,
                                   start_new_session=True)
    details = dict(pid=process.pid, started_utc=now(), command=command, threads=1)
    save(root/'nested_stage_process.json', details)
    print(json.dumps(details), flush=True)


if __name__ == '__main__':
    main()
