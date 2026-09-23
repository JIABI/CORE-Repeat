"""Bounded two-worker, resumable frozen-scatter ablation; no fit or R4 access."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opal2.m4_dependence_ablation import (DATASETS, REPORT_NAME, list_cells,
    run_cell, verify_dataset, write_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    output = PROJECT / REPORT_NAME
    if not (output / "PROTOCOL.md").exists():
        raise RuntimeError("The pre-run protocol must exist before execution")
    started = datetime.now(timezone.utc).isoformat()
    records = [r for r in list_cells(PROJECT) if r["dataset"] in args.datasets]
    write_json(output / "run_manifest.json", dict(started_utc=started, pid=os.getpid(),
        protocol="PROTOCOL.md", cells=records, workers=args.workers, samples=100000,
        arm="DIAGONAL_SCATTER", comparator="cached CORE_ORIGINAL", fit_performed=False,
        recalibration_performed=False, R4_accessed=False, endpoint_cost_per_well=0.01))
    for dataset in args.datasets:
        path = output / "verification" / f"{dataset}.json"
        if path.exists() and json.loads(path.read_text())["passed"]:
            print(f"VERIFIED CACHE {dataset}", flush=True)
        else:
            print(f"VERIFY {dataset}", flush=True)
            result = verify_dataset(PROJECT, dataset, output)
            print(f"VERIFIED {dataset}: maxdiff={max(result['max_absolute_differences'].values()):.3g}", flush=True)
    if args.verify_only:
        return
    begin = time.monotonic()
    complete = []
    write_json(output / "status.json", dict(state="RUNNING", started_utc=started,
               pid=os.getpid(), completed=0, total=len(records)))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_cell, PROJECT, record, output): record for record in records}
            for future in as_completed(futures):
                result = future.result()
                complete.append(result)
                print(f"COMPLETE {len(complete)}/{len(records)} {result['record']['label']} "
                      f"score_seconds={result['new_diagonal_score_seconds']:.1f}", flush=True)
                write_json(output / "status.json", dict(state="RUNNING", started_utc=started,
                    pid=os.getpid(), completed=len(complete), total=len(records),
                    elapsed_seconds=time.monotonic()-begin, latest=result["record"]["label"]))
        from opal2.m4_dependence_summary import summarize
        summarize(PROJECT, output, args.datasets)
        write_json(output / "status.json", dict(state="COMPLETE", started_utc=started,
            ended_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
            completed=len(complete), total=len(records), elapsed_seconds=time.monotonic()-begin))
    except Exception:
        write_json(output / "status.json", dict(state="FAILED", started_utc=started,
            pid=os.getpid(), completed=len(complete), total=len(records), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
