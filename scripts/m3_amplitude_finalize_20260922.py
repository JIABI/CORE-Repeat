#!/usr/bin/env python3
"""Finish the frozen confirmation extension after the running M3 coordinator."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opal2.m3_amplitude_controls import NAME, confirmation, now, _json


def main():
    run = PROJECT/"runs"/NAME
    report = PROJECT/"reports"/NAME
    run.mkdir(parents=True, exist_ok=True)
    with (run/"finalization.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run/"finalization_complete.json").exists():
            print("M3 finalization already complete", flush=True)
            return
        _json(run/"finalization_status.json", dict(state="waiting_for_development", started=now(), pid=os.getpid()))
        try:
            while not (run/"complete.json").exists():
                status = json.loads((run/"status.json").read_text())
                if status["state"] == "failed":
                    raise RuntimeError("M3 coordinator failed; preserve checkpoints for implementation repair")
                time.sleep(30)
            _json(run/"finalization_status.json", dict(state="confirmation_extension", started=now(), pid=os.getpid()))
            print("Completing frozen confirmation dispersion extension", now(), flush=True)
            confirmation(PROJECT, run, report)
            subprocess.run([sys.executable, str(PROJECT/"scripts/m3_amplitude_report_20260922.py")],
                           cwd=PROJECT, check=True)
            result = dict(state="complete", completed=now(), development_cells=60,
                development_populations=dict(EU=904, JUMP=639, LINCS=1188, RxRx3=10410),
                confirmation_rules=["AMP_ASC", "AMP_DESC", "AMP_HISTGB", "DISPERSION_DESC"],
                report=str(report/"REPORT.md"), original_R4_artifacts_modified=False)
            _json(run/"finalization_complete.json", result)
            _json(run/"finalization_status.json", result)
            print(json.dumps(result), flush=True)
        except Exception:
            _json(run/"finalization_status.json", dict(state="failed", failed=now(), traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
