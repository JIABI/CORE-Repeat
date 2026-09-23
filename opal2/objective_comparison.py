"""Frozen, full-configuration A/B/C experiment on the existing source_5 DEV split."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback

from .config import TrainConfig
from .data import load_dataset
from .splits import load_split


ARMS = ("A_ELBO", "B_NLL", "C_NLL_CRPS")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(root, state, **fields):
    payload = {"utc": now(), "state": state, **fields}
    temporary = root / "status.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(root / "status.json")
    print(json.dumps(payload), flush=True)


def configurations():
    base = TrainConfig(paired_objective_rng=True)
    return {
        "A_ELBO": replace(base, objective="elbo", utility_crps_weight=0.0).validate(),
        "B_NLL": replace(base, objective="predictive_nll", utility_crps_weight=0.0).validate(),
        "C_NLL_CRPS": replace(base, objective="predictive_nll", utility_crps_weight=1.0,
                              utility_crps_samples=16).validate(),
    }


def _load_declared_data(data):
    dataset = load_dataset(data / "measurements.npz")
    splits, scope = load_split(data / "splits.json", dataset.ids)
    counts = {name: len(ix) for name, ix in splits.items()}
    expected = {"train": 383, "validation": 96, "calibration": 64, "evaluation": 96}
    if dataset.Y.shape != (639, 4, 3617) or counts != expected:
        raise ValueError("This experiment requires the existing full 639-object/four-role DEV split")
    return dataset, splits, scope


def prepare(data, root):
    """Freeze the declared protocol/configuration/source before any arm is fitted."""
    data, root = Path(data).resolve(), Path(root).resolve()
    dataset, splits, scope = _load_declared_data(data)
    configs = configurations()
    root.mkdir(parents=True, exist_ok=False)
    (root / "arms").mkdir()
    project = Path(__file__).resolve().parents[1]
    shutil.copy2(project / "protocols/historical/OBJECTIVE_COMPARISON_PLAN.md", root / "PROTOCOL.md")
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    shutil.copytree(project / "opal2", snapshot / "opal2",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(project / "tests", snapshot / "tests",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(project / "pyproject.toml", snapshot / "pyproject.toml")
    shutil.copy2(project / "protocols/historical/OBJECTIVE_IMPLEMENTATION_CHECKS.md", root / "IMPLEMENTATION_CHECKS.md")
    shutil.copy2(data / "splits.json", root / "splits.json")
    for name, config in configs.items():
        config.save(root / (name + ".json"))
    import numpy as np
    import torch
    manifest = {
        "created_utc": now(), "purpose": "FULL_MODEL_TRAINING_OBJECTIVE_DEV_COMPARISON",
        "data_directory": str(data), "data_shape": list(dataset.Y.shape),
        "split_scope": scope, "compound_ids": {k: dataset.ids[v].tolist() for k, v in splits.items()},
        "configurations": {name: asdict(value) for name, value in configs.items()},
        "arm_order": list(ARMS), "evaluation_after_all_fitting": True,
        "shared_jepa_donor": "A_ELBO", "old_final_opened": False, "fifth_repeat_opened": False,
        "source_snapshot": str(snapshot), "python": sys.version,
        "python_executable": sys.executable, "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__,
        "training_seed_count": 1,
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_status(root, "PREPARED", output=str(root), split_counts={k: len(v) for k, v in splits.items()})
    return manifest


def execute(root, *, resume=False):
    """Fit all arms before opening their development evaluation outputs."""
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if Path(__file__).resolve().parents[1] != Path(manifest["source_snapshot"]):
        raise ValueError("Execute from the frozen source_snapshot directory, not the live working source")
    from .training import fit_model, load_model, seed_everything
    from .evaluation import evaluate_model
    from .objective_analysis import generate_analysis
    data = Path(manifest["data_directory"])
    dataset, splits, _ = _load_declared_data(data)
    if {k: dataset.ids[v].tolist() for k, v in splits.items()} != manifest["compound_ids"]:
        raise ValueError("Dataset/split identities no longer match the frozen experiment")
    configs = {name: TrainConfig.load(root / (name + ".json")) for name in ARMS}
    if {name: asdict(value) for name, value in configs.items()} != manifest["configurations"]:
        raise ValueError("Run configurations differ from the frozen manifest")
    try:
        for name in ARMS:
            destination = root / "arms" / name
            marker = destination / "fit_complete.json"
            if marker.exists():
                if not resume:
                    raise FileExistsError("Existing fitted arm; use explicit resume")
                continue
            if destination.exists() and not resume:
                raise FileExistsError("Existing unfinished arm; use explicit resume")
            can_resume = (destination / "last.pt").exists()
            if destination.exists() and not can_resume:
                raise RuntimeError("An interrupted pre-probability stage needs explicit diagnosis before restart")
            write_status(root, "TRAINING", arm=name, pid=os.getpid())
            model, scaler = fit_model(dataset, splits, configs[name], destination,
                                      resume=can_resume,
                                      shared_jepa_directory=(root / "arms" / "A_ELBO")
                                      if name != "A_ELBO" and not can_resume else None)
            _, _, _, checkpoint = load_model(destination)
            marker.write_text(json.dumps({"completed_utc": now(), "best_epoch": checkpoint["epoch"],
                                           "validation_score": checkpoint["best_validation"]}, indent=2) + "\n")
            del model, scaler, checkpoint
            gc.collect()
        import torch
        initial = torch.load(root / "arms" / ARMS[0] / "initial_state.pt", weights_only=True)
        for name in ARMS[1:]:
            other = torch.load(root / "arms" / name / "initial_state.pt", weights_only=True)
            if (initial.keys() != other.keys() or
                    any(not torch.equal(initial[key], other[key]) for key in initial)):
                raise ValueError("Arm initial states differ; paired objective comparison is invalid")
            del other
        (root / "initialization_comparison.json").write_text(json.dumps({
            "checked_utc": now(), "all_state_tensors_exactly_equal": True,
            "arms": list(ARMS), "tensor_count": len(initial),
        }, indent=2) + "\n")
        del initial
        for name in ARMS:
            destination = root / "arms" / name
            marker = destination / "evaluation_complete.json"
            if marker.exists() and resume:
                continue
            write_status(root, "EVALUATING", arm=name, pid=os.getpid())
            model, scaler, config, checkpoint = load_model(destination)
            seed_everything(config.seed, config.threads)
            evaluate_model(model, scaler, dataset, splits, config, destination)
            marker.write_text(json.dumps({"completed_utc": now(), "best_epoch": checkpoint["epoch"]}) + "\n")
            del model, scaler, checkpoint
            gc.collect()
        write_status(root, "ANALYZING", pid=os.getpid())
        report = generate_analysis(root)
        write_status(root, "COMPLETE", report=str(root / "RESULT_ANALYSIS.md"))
        return report
    except BaseException as exc:
        write_status(root, "FAILED", error_type=type(exc).__name__, error=str(exc),
                     traceback=traceback.format_exc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", required=True)
    prep.add_argument("--output", required=True)
    run = sub.add_parser("execute")
    run.add_argument("--output", required=True)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.data, args.output)
    else:
        execute(args.output, resume=args.resume)


if __name__ == "__main__":
    main()
