"""Read-only common-stopping check for the matched J/M/F DEV experiment.

This module reads validation JSON only. It never loads data or checkpoints,
writes output files, or stops a worker; STOP_ELIGIBLE is a recommendation for
the separate controller to act on at a complete matched checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence


ARMS = ("J_JOINT", "M_MEAN_ONLY", "F_MEAN_PROTECTED")
METRICS = ("physical", "standardized")
CHECK_INTERVAL = 5
MINIMUM_EPOCH = 30
PATIENCE_CHECKS = 4
MINIMUM_RELATIVE_IMPROVEMENT = 0.005
EXPECTED_N = 96


def _result(status, latest_epoch=None, per_metric=None, *, reason, count=0):
    return {
        "decision": status,
        "eligible": status == "STOP_ELIGIBLE",
        "latest_epoch": latest_epoch,
        "validation_count": count,
        "per_metric": per_metric or {},
        "reason": reason,
        "rule": {
            "minimum_epoch": MINIMUM_EPOCH,
            "check_interval": CHECK_INTERVAL,
            "patience_checks": PATIENCE_CHECKS,
            "minimum_relative_improvement": MINIMUM_RELATIVE_IMPROVEMENT,
            "expected_n": EXPECTED_N,
        },
    }


def evaluate_records(records: Sequence[Mapping]) -> dict:
    """Evaluate chronologically ordered, complete validation records.

    Each metric keeps its last significant-improvement anchor. Smaller
    improvements do not move that anchor, so their cumulative improvement
    can eventually reach the 0.5% threshold. A zero anchor cannot improve.
    """
    states = {arm: {} for arm in ARMS}
    latest = None
    if not records:
        return _result("NEEDS_REVIEW", reason="No validation records exist")
    for index, record in enumerate(records):
        expected_epoch = (index + 1) * CHECK_INTERVAL
        try:
            if not isinstance(record, Mapping):
                raise ValueError("Validation record must be an object")
            epoch = record["epoch"]
            if type(epoch) is not int or epoch != expected_epoch:
                raise ValueError(f"Expected consecutive epoch {expected_epoch}, got {epoch!r}")
            latest = epoch
            arms = record["arms"]
            if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
                raise ValueError("Validation must contain exactly the three matched arms")
            values = {}
            for arm in ARMS:
                report = arms[arm]
                if type(report["n"]) is not int or report["n"] != EXPECTED_N:
                    raise ValueError(f"{arm} must have n={EXPECTED_N}")
                values[arm] = {}
                for metric in METRICS:
                    value = report["measurement"][metric]["overall"]["mse"]
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"{arm}/{metric} MSE must be finite and nonnegative")
                    values[arm][metric] = float(value)
            for metric in METRICS:
                if values["M_MEAN_ONLY"][metric] != values["F_MEAN_PROTECTED"][metric]:
                    raise ValueError(f"M/F {metric} MSE differs")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            return _result("NEEDS_REVIEW", latest, states,
                           reason=f"Invalid validation record: {error}", count=index)

        for arm in ARMS:
            for metric in METRICS:
                value = values[arm][metric]
                state = states[arm].get(metric)
                if state is None:
                    states[arm][metric] = {
                        "anchor": value, "last_significant_epoch": epoch,
                        "stale_checks": 0, "latest_mse": value,
                    }
                    continue
                improved = state["anchor"] > 0 and value <= (
                    state["anchor"] * (1 - MINIMUM_RELATIVE_IMPROVEMENT))
                if improved:
                    state.update(anchor=value, last_significant_epoch=epoch, stale_checks=0)
                else:
                    state["stale_checks"] += 1
                state["latest_mse"] = value

    eligible = latest >= MINIMUM_EPOCH and all(
        state["stale_checks"] >= PATIENCE_CHECKS
        for metrics in states.values() for state in metrics.values())
    reason = (
        "All three arms and both metrics meet the common validation-stagnation rule"
        if eligible else
        "Minimum epoch or simultaneous four-check stagnation has not been reached"
    )
    return _result("STOP_ELIGIBLE" if eligible else "CONTINUE", latest, states,
                   reason=reason, count=len(records))


def inspect_run(output: str | Path) -> dict:
    """Read only RUN/validation/epoch_*.json; reject gaps or malformed files."""
    folder = Path(output) / "validation"
    try:
        paths = list(folder.glob("epoch_*.json"))
        indexed = []
        for path in paths:
            match = re.fullmatch(r"epoch_(\d+)\.json", path.name)
            if match is None:
                raise ValueError(f"Unexpected validation filename: {path.name}")
            indexed.append((int(match.group(1)), path))
        records = []
        for epoch, path in sorted(indexed):
            record = json.loads(path.read_text())
            if not isinstance(record, Mapping) or record.get("epoch") != epoch:
                raise ValueError(f"Filename and record epoch differ: {path.name}")
            records.append(record)
        return evaluate_records(records)
    except (OSError, ValueError, TypeError, OverflowError) as error:
        return _result("NEEDS_REVIEW", reason=f"Cannot validate JSON history: {error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Existing run directory")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_run(args.output), allow_nan=False))


if __name__ == "__main__":
    main()
