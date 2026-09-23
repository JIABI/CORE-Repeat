#!/usr/bin/env python3
"""Run all prespecified M3 controls with at most two CPU workers."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import sklearn
from opal2.m3_amplitude_controls import NAME, now, fit_cell, aggregate_development, confirmation, _json
from opal2.quantile_direct_data import list_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate", "confirmation"))
    args = parser.parse_args()
    run = PROJECT/"runs"/NAME
    report = PROJECT/"reports"/NAME
    run.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    protocol = PROJECT/"protocols/m3_amplitude_controls_20260922.md"
    if sklearn.__version__ != "1.9.1":
        raise RuntimeError("Protocol specifies scikit-learn 1.9.1")
    if args.command == "confirmation":
        confirmation(PROJECT, run, report)
        return
    if args.command == "aggregate":
        aggregate_development(PROJECT, run, report)
        return
    if (run/"complete.json").exists():
        raise RuntimeError("M3 already complete; use aggregate only for reporting changes")
    records = list_cells(PROJECT)
    manifest = run/"run_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["cells"] != records or old["protocol_text"] != protocol.read_text():
            raise ValueError("Run manifest or protocol changed")
    else:
        _json(manifest, dict(started=now(), cells=records, protocol_text=protocol.read_text(),
            sklearn=sklearn.__version__, workers=2, numerical_threads_per_worker=1, gpu=False))
    try:
        _json(run/"status.json", dict(state="running", started=now(), completed_cells=0, total_cells=60))
        print("M3 full execution started", now(), flush=True)
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(fit_cell, str(PROJECT), str(run), record) for record in records]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                _json(run/"status.json", dict(state="running", updated=now(), completed_cells=i,
                                             total_cells=60, latest=result))
                print(json.dumps(result), flush=True)
        print("M3 development aggregation", now(), flush=True)
        summary = aggregate_development(PROJECT, run, report)
        print("M3 post-hoc confirmation", now(), flush=True)
        confirmation(PROJECT, run, report)
        _json(run/"complete.json", dict(completed=now(), state="complete", cells=60, report=str(report)))
        _json(run/"status.json", dict(completed=now(), state="complete", cells=60, report=str(report)))
        print(json.dumps(summary), flush=True)
    except Exception:
        _json(run/"status.json", dict(state="failed", failed=now(), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
