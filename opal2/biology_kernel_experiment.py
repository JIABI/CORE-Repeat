"""Complete four-arm, three-seed biology-kernel/observation DEV experiment.

Preparation records the entire queue and source before optimization. Each job
is a full configured world model. Partial summaries never change the remaining
queue or nominate a winner. Explicit resume restores training checkpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .biology_kernel_evaluation import plain, write_json
from .config import TrainConfig


SEEDS = (20260912, 20260913, 20260914)
ARM_FACTORS = {
    "A_OFF_GAUSSIAN": ("off", "gaussian"),
    "B_OFF_COPULA_T4": ("off", "copula_t4"),
    "C_KERNEL_GAUSSIAN": ("structured", "gaussian"),
    "D_KERNEL_COPULA_T4": ("structured", "copula_t4"),
}
SPLITS = ("validation", "calibration", "evaluation")


def now():
    return datetime.now(timezone.utc).isoformat()


def status(root, state, **fields):
    value = dict(utc=now(), state=state, **fields)
    with (root / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(plain(value), ensure_ascii=False) + "\n")
    write_json(root / "status.json", value)
    print(json.dumps(plain(value), ensure_ascii=False), flush=True)


def configurations():
    """Declared complete main-model configurations, not reduced smoke arms."""
    base = TrainConfig(epochs=100, batch_size=32, learning_rate=3e-4,
        lr_schedule="cosine", warmup_steps=60, min_learning_rate=3e-6,
        weight_decay=1e-4, patience=15, hidden_dim=256, latent_rank=32, residual_rank=8,
        group_attention_layers=2, attention_heads=4, kernel_mode="measurement",
        use_jepa=False, encoder_policy="trainable", use_chemistry=True,
        use_references=True, use_library=True, use_biology_prior=False,
        biology_kernel_anchors=64, objective="predictive_nll", utility_crps_weight=0.,
        paired_objective_rng=True, samples=2000, mc_chunk_size=32,
        reference_access="observed_only", device="cpu", threads=4)
    return {str(seed): {name: replace(base, seed=seed, biology_kernel_mode=kernel,
                                     observation_family=family).validate()
                       for name, (kernel, family) in ARM_FACTORS.items()} for seed in SEEDS}


def _load_study_data(data):
    from .objective_comparison import _load_declared_data
    data = Path(data).resolve()
    if data.name != "source5_primary_fullcontrols":
        raise ValueError("This protocol permits only the existing source5_primary_fullcontrols export")
    source = json.loads((data / "manifest.json").read_text())
    if source.get("old_final_opened") is not False or source.get("fifth_repeat_read") is not False:
        raise ValueError("Dataset manifest does not preserve original FINAL/fifth-repeat boundary")
    ds, split, scope = _load_declared_data(data)
    if not np.all(ds.observed_mask) or not np.all(ds.well_mask) or not np.isfinite(ds.Y).all():
        raise ValueError("The declared complete cohort changed; no automatic exclusions are permitted")
    return ds, split, scope


def prepare(data, root, protocol_path):
    """Create a fresh, frozen 12-job queue. Does not optimize any model."""
    data, root, protocol = Path(data).resolve(), Path(root).resolve(), Path(protocol_path).resolve()
    if root.exists():
        raise FileExistsError("Choose an entirely new run directory")
    if not protocol.is_file():
        raise FileNotFoundError("A written protocol is required before preparing this experiment")
    ds, splits, scope = _load_study_data(data)
    configs = configurations()
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    project = Path(__file__).resolve().parents[1]
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    shutil.copytree(project / "opal2", snapshot / "opal2", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(project / "tests", snapshot / "tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(project / "pyproject.toml", snapshot / "pyproject.toml")
    shutil.copy2(protocol, root / "PROTOCOL.md")
    queue = []
    for seed, arms in configs.items():
        for arm, cfg in arms.items():
            name = f"{seed}__{arm}"
            cfg.save(root / "configs" / f"{name}.json")
            queue.append(dict(seed=int(seed), arm=arm, config=f"configs/{name}.json",
                              directory=f"jobs/seed_{seed}/{arm}"))
    manifest = dict(created_utc=now(), purpose="FULL_BIOLOGY_KERNEL_NOISE_FACTORIAL_OPEN_DEV",
        data_directory=str(data), data_shape=list(ds.Y.shape), feature_names=ds.feature_names.tolist(),
        split_scope=scope, compound_ids={key: ds.ids[ix].tolist() for key, ix in splits.items()},
        source_snapshot=str(snapshot), configurations={s: {a: asdict(c) for a, c in arms.items()}
                                                    for s, arms in configs.items()},
        queue=queue, arm_factors=ARM_FACTORS, seeds=list(SEEDS),
        evaluation_splits=list(SPLITS), fractions=[.05, .10, .25],
        n_bootstrap=2000, n_random=2000, evaluation_object_chunk=2,
        original_endpoint_changed=False, original_contract_changed=False,
        final_opened=False, fifth_repeat_opened=False, mechanism_annotations_active=False,
        jepa_active=False, historical_dev=True, checkpoint_selection="validation fixed-role predictive joint NLL in fixed minibatches",
        reporting_score="proper per-compound marginal density, not identical to checkpoint block-joint score",
        results_do_not_modify_remaining_queue=True, python=sys.version)
    write_json(root / "run_manifest.json", manifest)
    status(root, "PREPARED", jobs=len(queue), full_configs=True,
           split_counts={key: len(ix) for key, ix in splits.items()})
    return manifest


def _load_bound_run(root, *, require_snapshot=True):
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if require_snapshot and Path(__file__).resolve().parents[1] != Path(manifest["source_snapshot"]):
        raise ValueError("Execute model fitting/evaluation from this run's frozen source_snapshot")
    if manifest.get("final_opened") is not False or manifest.get("fifth_repeat_opened") is not False:
        raise ValueError("Only the declared four-role DEV run is supported")
    if manifest.get("original_contract_changed") is not False or manifest.get("mechanism_annotations_active") is not False:
        raise ValueError("Experiment boundaries differ from the declared main comparison")
    configs = {}
    expected_queue = [(int(s), a) for s in manifest["seeds"] for a in ARM_FACTORS]
    if [(int(j["seed"]), j["arm"]) for j in manifest["queue"]] != expected_queue:
        raise ValueError("The complete seed/arm queue changed")
    for job in manifest["queue"]:
        cfg = TrainConfig.load(root / job["config"])
        expected = manifest["configurations"][str(job["seed"])][job["arm"]]
        if asdict(cfg) != expected:
            raise ValueError("Configuration differs from its frozen declaration")
        configs[(job["seed"], job["arm"])] = cfg
    ds, splits, _ = _load_study_data(manifest["data_directory"])
    if {key: ds.ids[ix].tolist() for key, ix in splits.items()} != manifest["compound_ids"]:
        raise ValueError("Dataset/split identities differ from the declared experiment")
    if ds.feature_names.tolist() != manifest["feature_names"]:
        raise ValueError("Fixed measurement coordinate order changed")
    return root, manifest, ds, splits, configs


def _compact_partition(report):
    return dict(n=report["n"], measurement=report["measurement"], proper_nll=report["proper_nll"],
        predictive_intervals=report["predictive_intervals"], mean_mc_se=report["mean_mc_se"],
        utility_crps_by_action=report["utility_crps_by_action"],
        action_metrics=report["policy"]["action_metrics"],
        within_action=[{key: row[key] for key in ("label", "action", "ranking", "fraction", "selected_n",
            "used_wells", "per_selected_net_gain", "per_eligible_net_gain", "fdp", "fpr", "sensitivity",
            "matched_random")} for row in report["policy"]["within_action"]],
        common_budget=[{key: row[key] for key in ("label", "action", "ranking", "fraction", "selected_n",
            "used_wells", "budget_wells", "per_selected_net_gain", "per_eligible_net_gain", "fdp", "fpr", "sensitivity")}
            for row in report["policy"]["common_budget"]],
        fixed_policies=report["policy"]["fixed_policies"],
        train_selected_fixed_action=report["policy"]["train_selected_fixed_action"])


def summarize(root):
    """Partial arm summaries; never selects a seed, arm, threshold, or budget."""
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    jobs, complete = [], 0
    for job in manifest["queue"]:
        directory = root / job["directory"]
        entry = dict(seed=job["seed"], arm=job["arm"], fit_complete=(directory / "fit_complete.json").exists(), partitions={})
        for split in SPLITS:
            path = directory / split / "metrics.json"
            if path.exists():
                report = json.loads(path.read_text())
                if report["ids"] != manifest["compound_ids"][split] or report.get("formal_certificate") is not False:
                    raise ValueError("Saved partition report does not match the declared evaluation population")
                if not (directory / split / "predictions.npz").exists():
                    raise ValueError("Completed metrics lack per-object evidence")
                entry["partitions"][split] = _compact_partition(report)
        entry["complete"] = entry["fit_complete"] and len(entry["partitions"]) == len(SPLITS)
        complete += int(entry["complete"])
        jobs.append(entry)
    result = dict(created_utc=now(), completed_jobs=complete, total_jobs=len(jobs),
                  complete=complete == len(jobs), jobs=jobs, seeds_are_not_extra_compounds=True,
                  result_based_queue_changes=False, historical_dev=True, final_opened=False,
                  fifth_repeat_opened=False, original_contract_changed=False, formal_certificate=False)
    from .biology_kernel_comparison import summarize_factorial
    comparisons = summarize_factorial(root)
    result["factorial_comparison_file"] = "factorial_comparison.json"
    result["average_seed_pending_contrasts"] = {split: values["pending"]
        for split, values in comparisons["average_seed"].items()}
    write_json(root / "summary.json", result)
    return result


def _check_common_initialization(root, manifest, job):
    import torch
    first = next(j for j in manifest["queue"] if j["seed"] == job["seed"])
    if first == job:
        return
    current_path = root / job["directory"] / "initial_state.pt"
    reference_path = root / first["directory"] / "initial_state.pt"
    current = torch.load(current_path, map_location="cpu", weights_only=True)
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    common = sorted(set(current) & set(reference))
    if not common or any(current[key].shape != reference[key].shape or not torch.equal(current[key], reference[key]) for key in common):
        raise ValueError("Common initial model tensors differ across paired arms")
    write_json(root / job["directory"] / "initialization_comparison.json",
               dict(reference_arm=first["arm"], seed=job["seed"], common_tensors=len(common),
                    common_tensors_exactly_equal=True, extra_tensors=sorted(set(current) - set(reference))))


def execute(root, *, resume=False, max_jobs=None):
    """Run complete jobs in the frozen order, with checkpoint-level resumption."""
    from .training import fit_model, load_model, seed_everything
    from .biology_kernel_evaluation import evaluate_partition
    root, manifest, ds, splits, configs = _load_bound_run(root)
    if max_jobs is not None and (isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 1):
        raise ValueError("max_jobs must be a positive count of complete jobs")
    completed_this_call = 0
    started = time.monotonic()
    try:
        for job in manifest["queue"]:
            directory = root / job["directory"]
            config = configs[(job["seed"], job["arm"])]
            fit_marker = directory / "fit_complete.json"
            eval_marker = directory / "evaluation_complete.json"
            if eval_marker.exists():
                if not resume:
                    raise FileExistsError("Completed job exists; use explicit resume")
                required = [fit_marker, directory / "best.pt"] + [directory / split / name
                    for split in SPLITS for name in ("metrics.json", "predictions.npz", "predictions.tsv")]
                if not all(path.is_file() for path in required):
                    raise RuntimeError("Completed job marker has missing fitted/evaluation artifacts")
                continue
            if fit_marker.exists():
                if not resume:
                    raise FileExistsError("Fitted job exists; use explicit resume")
            else:
                if directory.exists() and not resume:
                    raise FileExistsError("Existing job requires explicit resume")
                continuation = (directory / "last.pt").exists()
                if directory.exists() and not continuation:
                    raise RuntimeError("Existing unfinished job has no training checkpoint; inspect before restarting")
                status(root, "TRAINING", seed=job["seed"], arm=job["arm"], resume=continuation)
                model, scaler = fit_model(ds, splits, config, directory, resume=continuation)
                _, _, _, checkpoint = load_model(directory)
                _check_common_initialization(root, manifest, job)
                write_json(fit_marker, dict(utc=now(), best_epoch=checkpoint["epoch"],
                    best_validation_score=checkpoint["best_validation"],
                    selection="validation block-joint predictive NLL; not calibration/evaluation"))
                del model, scaler, checkpoint
                gc.collect()
            model, scaler, loaded, checkpoint = load_model(directory)
            if asdict(loaded) != asdict(config):
                raise ValueError("Loaded checkpoint configuration differs from frozen job")
            seed_everything(config.seed, config.threads)
            for split in SPLITS:
                part = directory / split
                if (part / "metrics.json").exists():
                    if not resume:
                        raise FileExistsError("Existing partition evaluation requires explicit resume")
                    continue
                status(root, "EVALUATING", seed=job["seed"], arm=job["arm"], split=split)
                evaluate_partition(model, scaler, ds, splits["train"], splits[split], config, part,
                    seed=config.seed, fractions=manifest["fractions"], object_chunk=manifest["evaluation_object_chunk"],
                    n_bootstrap=manifest["n_bootstrap"], n_random=manifest["n_random"],
                    progress=lambda done, count: status(root, "EVALUATING", seed=job["seed"],
                        arm=job["arm"], split=split, done=done, n=count))
                summarize(root)
            write_json(eval_marker, dict(utc=now(), best_epoch=checkpoint["epoch"], splits=list(SPLITS)))
            del model, scaler, checkpoint
            gc.collect()
            completed_this_call += 1
            summary = summarize(root)
            status(root, "JOB_COMPLETE", seed=job["seed"], arm=job["arm"],
                   completed_jobs=summary["completed_jobs"], total_jobs=summary["total_jobs"],
                   elapsed_seconds=time.monotonic() - started)
            if max_jobs is not None and completed_this_call >= max_jobs:
                break
        result = summarize(root)
        status(root, "COMPLETE" if result["complete"] else "PAUSED_COMPLETE_JOB_BOUNDARY",
               completed_jobs=result["completed_jobs"], total_jobs=result["total_jobs"])
        return result
    except BaseException as error:
        status(root, "FAILED", error_type=type(error).__name__, error=str(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--protocol", required=True)
    run = sub.add_parser("execute")
    run.add_argument("--output", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-jobs", type=int)
    summary = sub.add_parser("summarize")
    summary.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.data, args.output, args.protocol)
    elif args.command == "execute":
        execute(args.output, resume=args.resume, max_jobs=args.max_jobs)
    else:
        result = summarize(args.output)
        print(json.dumps(dict(completed_jobs=result["completed_jobs"], total_jobs=result["total_jobs"],
                              complete=result["complete"])))


if __name__ == "__main__":
    main()
