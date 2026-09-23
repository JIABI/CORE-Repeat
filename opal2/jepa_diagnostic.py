"""Prepare, run and inspect JEPA-only optimization diagnostics on existing DEV."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import traceback


def now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(data, output):
    from .config import TrainConfig
    from .data import TrainScaler, fit_library_context, attach_library_context
    from .objective_comparison import _load_declared_data
    from .training import model_kwargs
    from .jepa_diagnostic_training import DiagnosticConfig

    data, root = Path(data).resolve(), Path(output).resolve()
    dataset, splits, scope = _load_declared_data(data)
    training = TrainConfig(batch_size=128).validate()
    diagnostic = DiagnosticConfig().validate()
    root.mkdir(parents=True, exist_ok=False)
    training.save(root / "train_config.json")
    _write_json(root / "diagnostic_config.json", asdict(diagnostic))
    scaler = TrainScaler.fit(dataset, splits["train"])
    scaler.save(root / "scaler.json")
    normalized = scaler.transform(dataset)
    bank = fit_library_context(normalized, splits["train"])
    bank.save(root / "library_context.npz")
    normalized = attach_library_context(normalized, bank)
    project = Path(__file__).resolve().parents[1]
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for name in ("opal2", "tests"):
        shutil.copytree(project / name, snapshot / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(project / "pyproject.toml", snapshot / "pyproject.toml")
    shutil.copy2(project / "protocols/historical/JEPA_DIAGNOSTIC_PLAN.md", root / "PROTOCOL.md")
    shutil.copy2(data / "splits.json", root / "splits.json")
    manifest = {
        "created_utc": now(), "purpose": "JEPA_OPTIMIZATION_DIAGNOSTIC_ONLY",
        "data_directory": str(data), "source_snapshot": str(snapshot),
        "scope": scope, "data_shape": list(dataset.Y.shape),
        "train_config": asdict(training), "diagnostic_config": asdict(diagnostic),
        "model_kwargs": model_kwargs(normalized, training),
        "compound_ids": {k: dataset.ids[v].tolist() for k, v in splits.items()},
        "used_partitions": ["train", "validation"],
        "existing_dev_archive_loaded": True,
        "calibration_outcomes_used": False, "evaluation_outcomes_used": False,
        "formal_training_suspended_pid": 89675,
        "formal_run": str(project / "runs/objective_comparison_primary_20260911_v1"),
        "old_final_accessed": False, "fifth_repeat_accessed": False,
        "automatic_world_model_training": False,
        "python_executable": sys.executable,
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "status.json", {"utc": now(), "state": "PREPARED"})
    return manifest


def execute(output):
    root = Path(output).resolve()
    try:
        return _execute(root)
    except BaseException as error:
        _write_json(root / "status.json", {
            "utc": now(), "state": "FAILED", "phase": "diagnostic_execute",
            "pid": os.getpid(), "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
        })
        raise


def _execute(output):
    from .config import TrainConfig
    from .data import TrainScaler, LibraryBank, attach_library_context
    from .objective_comparison import _load_declared_data
    from .jepa_diagnostic_training import DiagnosticConfig, run_diagnostics

    root = Path(output).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if Path(__file__).resolve().parents[1] != Path(manifest["source_snapshot"]):
        raise ValueError("Run from this diagnostic's source_snapshot")
    training = TrainConfig.load(root / "train_config.json")
    diagnostic = DiagnosticConfig(**json.loads((root / "diagnostic_config.json").read_text())).validate()
    if json.loads(json.dumps(asdict(diagnostic))) != manifest["diagnostic_config"]:
        raise ValueError("Diagnostic settings changed after preparation")
    if asdict(training) != manifest["train_config"]:
        raise ValueError("Training settings changed after preparation")
    dataset, splits, _ = _load_declared_data(Path(manifest["data_directory"]))
    if {k: dataset.ids[v].tolist() for k, v in splits.items()} != manifest["compound_ids"]:
        raise ValueError("Compound partitions changed")
    scaler = TrainScaler.load(root / "scaler.json")
    if scaler.train_ids != manifest["compound_ids"]["train"]:
        raise ValueError("Scaler does not match the training partition")
    normalized = attach_library_context(scaler.transform(dataset),
                                       LibraryBank.load(root / "library_context.npz"))
    return run_diagnostics(normalized, splits, training, manifest["model_kwargs"], root,
                           diagnostic_config=diagnostic)


def inspect(output, *, compact=False):
    root = Path(output).resolve()
    result = {"output": str(root)}
    for name in ("status.json", "diagnostic_summary.json"):
        if (root / name).exists():
            result[name.removesuffix(".json")] = json.loads((root / name).read_text())
    result["logs"] = {}
    latest_epochs, latest_diagnostics = {}, {}
    # Logs only: no profile archive, checkpoint, or outcome partition is loaded.
    for path in sorted(root.rglob("*.jsonl")):
        if "source_snapshot" in path.parts:
            continue
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A writer may be appending its current final line.
        if rows:
            result["logs"][str(path.relative_to(root))] = rows[-2:]
        for row in rows:
            if row.get("event") == "epoch" and "arm" in row:
                latest_epochs[row["arm"]] = row
            if row.get("event") == "diagnostic" and "arm" in row:
                latest_diagnostics[row["arm"]] = row
    if compact:
        result.pop("logs")
        result["latest_epochs"] = latest_epochs
        result["latest_diagnostics"] = {}
        for arm, row in latest_diagnostics.items():
            brief = {key: row[key] for key in ("utc", "epoch") if key in row}
            for split in ("train", "validation"):
                metrics = row.get(split, {})
                brief[split] = {key: metrics.get(key) for key in (
                    "alignment_mse", "teacher_target_variance", "alignment_nmse",
                    "alignment_cosine", "zero_teacher_target_variance")}
                brief[split]["representations"] = {
                    name: {key: values.get(key) for key in (
                        "std_mean", "std_min", "near_zero_dimension_fraction",
                        "covariance_participation_rank", "covariance_entropy_rank")}
                    for name, values in metrics.get("representations", {}).items()
                }
            brief["gradient_components"] = row.get("gradient_components")
            result["latest_diagnostics"][arm] = brief
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", required=True)
    prep.add_argument("--output", required=True)
    run = sub.add_parser("execute")
    run.add_argument("--output", required=True)
    status = sub.add_parser("inspect")
    status.add_argument("--output", required=True)
    status.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare(args.data, args.output)
        result = {"state": "PREPARED", "output": str(Path(args.output).resolve()),
                  "data_shape": manifest["data_shape"],
                  "split_counts": {k: len(v) for k, v in manifest["compound_ids"].items()},
                  "diagnostic_config": manifest["diagnostic_config"]}
    elif args.command == "execute":
        result = execute(args.output)
    else:
        result = inspect(args.output, compact=args.compact)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
