"""Full-coordinate parameter-refit diagnostics for the frozen first-seed A.

No backbone is retrained here. The complete fitted Gaussian anchor and the
two explicit output-distribution repairs are saved and evaluated on the
unchanged historical DEV partitions, using original utilities and budgets.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr

from .biology_kernel_evaluation import (actual_gains, cosine, evaluate_partition,
                                      score_measurements, write_json)


ARMS = ("L_FULL_GAUSSIAN", "A_MEAN_ANCHOR", "A_MEAN_WITHIN_ANCHOR")
PARTITIONS = ("validation", "evaluation", "calibration")


def now():
    return datetime.now(timezone.utc).isoformat()


def status(root, state, **fields):
    item = dict(utc=now(), state=state, **fields)
    write_json(root / "status.json", item)
    with (root / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(item) + "\n")
    print(json.dumps(item), flush=True)


def process_identity(pid):
    return subprocess.check_output(
        ["ps", "-p", str(pid), "-o", "pid=,lstart=,command="], text=True).strip()


def prepare(root, source_run, paused_pid=None):
    from .biology_kernel_experiment import _load_study_data
    root, source_run = Path(root).resolve(), Path(source_run).resolve()
    if root.exists():
        raise FileExistsError("Use a fresh repair directory")
    manifest = json.loads((source_run / "run_manifest.json").read_text())
    data = Path(manifest["data_directory"])
    ds, split, scope = _load_study_data(data)
    if {k: ds.ids[v].tolist() for k, v in split.items()} != manifest["compound_ids"]:
        raise ValueError("Repair objects differ from the frozen experiment")
    checkpoint = source_run / "jobs/seed_20260912/A_OFF_GAUSSIAN"
    if not (checkpoint / "evaluation_complete.json").is_file():
        raise ValueError("The declared seed1 A is not complete")
    root.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    shutil.copy2(project / "protocols/historical/MOMENT_REPAIR_PLAN_20260913.md", root / "PROTOCOL.md")
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    shutil.copytree(project / "opal2", snapshot / "opal2",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(project / "tests", snapshot / "tests",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    pause = None
    if paused_pid is not None:
        identity = process_identity(paused_pid)
        if "opal2.biology_kernel_experiment execute" not in identity or str(source_run) not in identity:
            raise ValueError("Only the verified old factorial process may be resumed")
        pause = dict(pid=paused_pid, identity=identity)
    result = dict(created_utc=now(), source_run=str(source_run), checkpoint=str(checkpoint),
        data_directory=str(data), compound_ids=manifest["compound_ids"], scope=scope,
        arms=list(ARMS), partitions=list(PARTITIONS), seed=20260912,
        fit=dict(k=200, clip=8., noise_shrinkage=.05, variance_floor=1e-6,
                 random_state=20260912),
        samples=2000, mc_chunk_size=32, object_chunk=2, threads=4,
        fractions=manifest["fractions"], n_bootstrap=2000, n_random=2000,
        source_snapshot=str(snapshot), paused_process=pause,
        neural_gradient_updates=0, full_measurement_dimension=3617,
        old_results_modified=False, final_opened=False, fifth_repeat_opened=False,
        original_endpoint_changed=False, original_contract_changed=False,
        historical_dev=True, formal_certificate=False)
    write_json(root / "run_manifest.json", result)
    status(root, "PREPARED")


class GeometryDistribution:
    """Record full simulated-well geometry without retaining giant profiles."""
    def __init__(self, distribution, scaler):
        self.distribution = distribution
        self.scaler = scaler
        mean = scaler.inverse_y(distribution.mean).detach().cpu().numpy()
        mean_cosine = np.column_stack([cosine(mean[:, i], mean[:, j])
                                       for i, j in ((0, 1), (0, 2), (1, 2))])
        self.record = SimpleNamespace(mean_cosine=mean_cosine,
            sum_cosine=np.zeros_like(mean_cosine), sum_cosine_square=np.zeros_like(mean_cosine),
            sum_rms=np.zeros(mean.shape[:2]), count=0)

    def __getattr__(self, name):
        return getattr(self.distribution, name)

    def sample_joint(self, n_samples, generator=None, environment_noise_cache=None):
        value = self.distribution.sample_joint(n_samples, generator,
            environment_noise_cache=environment_noise_cache)
        raw = self.scaler.inverse_y(value).detach().cpu().numpy()
        cosines = np.stack([cosine(raw[:, :, i], raw[:, :, j])
                           for i, j in ((0, 1), (0, 2), (1, 2))], -1)
        self.record.sum_cosine += cosines.sum(0)
        self.record.sum_cosine_square += np.square(cosines).sum(0)
        self.record.sum_rms += np.sqrt(np.square(raw).mean(-1)).sum(0)
        self.record.count += n_samples
        return value


class GeometryModel(torch.nn.Module):
    def __init__(self, model, scaler):
        super().__init__()
        self.model, self.scaler = model, scaler
        self.records = []
        if hasattr(model, "library_bank"):
            self.library_bank = model.library_bank

    def forward(self, batch):
        item = GeometryDistribution(self.model(batch), self.scaler)
        # Only small accumulators survive a chunk; retaining each distribution
        # would retain its full coordinate covariance factors for the partition.
        self.records.append(item.record)
        return item

    def save_geometry(self, path, actual):
        if not self.records or any(r.count < 2 for r in self.records):
            raise ValueError("Geometry requires complete joint samples")
        means = np.concatenate([r.mean_cosine for r in self.records])
        sample_mean = np.concatenate([r.sum_cosine / r.count for r in self.records])
        sample_second = np.concatenate([r.sum_cosine_square / r.count for r in self.records])
        sample_rms = np.concatenate([r.sum_rms / r.count for r in self.records])
        realised = np.column_stack([cosine(actual[:, 1+i], actual[:, 1+j])
                                   for i, j in ((0, 1), (0, 2), (1, 2))])
        if len(means) != len(actual):
            raise ValueError("Geometry records differ from evaluated objects")
        report = dict(pairs=["Z1-Z2", "Z1-V", "Z2-V"],
            mean_vector_cosine=means.mean(0).tolist(),
            full_sample_cosine_mean=sample_mean.mean(0).tolist(),
            full_sample_cosine_sd=np.sqrt(np.maximum(0,
                sample_second.mean(0)-sample_mean.mean(0)**2)).tolist(),
            realised_well_cosine_mean=realised.mean(0).tolist(),
            full_sample_rms_mean=sample_rms.mean(0).tolist(),
            realised_rms_mean=np.sqrt(np.square(actual[:, 1:]).mean(-1)).mean(0).tolist(),
            comparison="conditional predictive samples pooled over the fixed observed X values; mean-vector cosine is separate")
        write_json(path / "geometry.json", report)
        np.savez_compressed(path / "geometry.npz", mean_vector_cosine=means,
            sample_cosine_mean=sample_mean, sample_cosine_second_moment=sample_second,
            realised_cosine=realised, sample_rms=sample_rms)
        self.records.clear()


def _number(x):
    return float(x) if np.isfinite(x) else None


def summarize(root, manifest):
    result = dict(updated_utc=now(), partitions={}, historical_dev=True,
                  formal_certificate=False, neural_gradient_updates=0)
    for part in PARTITIONS:
        reports = {}
        for arm in ("A_ORIGINAL",) + ARMS:
            folder = (Path(manifest["checkpoint"]) / part if arm == "A_ORIGINAL"
                      else root / "arms" / arm / part)
            if not (folder / "metrics.json").exists():
                continue
            metric = json.loads((folder / "metrics.json").read_text())
            with np.load(folder / "predictions.npz", allow_pickle=False) as a:
                if a["ids"].tolist() != manifest["compound_ids"][part]:
                    raise ValueError("Saved repair population changed")
                reports[arm] = dict(
                    measurement_mse=metric["measurement"]["physical"]["overall"]["mse"],
                    measurement_r2=metric["measurement"]["physical"]["overall"]["r2_training_mean"],
                    nll=metric["proper_nll"]["original_space_per_coordinate"],
                    intervals95=next(x for x in metric["predictive_intervals"] if x["level"] == .95),
                    predicted_gain=a["predicted"].mean(0).tolist(),
                    actual_gain=a["actual"].mean(0).tolist(),
                    predicted_null=a["p_null"].mean(0).tolist(),
                    actual_null=(a["actual"] <= 0).mean(0).tolist(),
                    spearman=[_number(spearmanr(a["predicted"][:, j], a["actual"][:, j]).statistic)
                              for j in range(3)],
                    crps=metric["utility_crps_by_action"],
                    common_budget=metric["policy"]["common_budget"])
        result["partitions"][part] = reports
    result["complete"] = all(len(v) == 4 for v in result["partitions"].values())
    write_json(root / "summary.json", result)
    return result


def restore_paused_process(root, manifest):
    pause = manifest.get("paused_process")
    if not pause:
        return
    try:
        current = process_identity(pause["pid"])
        if current != pause["identity"]:
            raise RuntimeError("Paused process identity changed; refusing to signal another process")
        os.kill(pause["pid"], signal.SIGCONT)
        write_json(root / "resource_restoration.json", dict(utc=now(), restored=True, **pause))
    except Exception as error:
        write_json(root / "resource_restoration.json", dict(utc=now(), restored=False,
                                                          error=str(error), **pause))
        raise


def execute(root):
    from .biology_kernel_experiment import _load_study_data
    from .closed_form_baseline import fit_baseline
    from .moment_repair import ClosedFormGaussianAdapter, MomentRepairModel
    from .training import load_model, seed_everything
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if str(Path(__file__).resolve().parents[1]) != manifest["source_snapshot"]:
        raise ValueError("Execute the recorded source snapshot")
    started = time.monotonic()
    try:
        ds, split, _ = _load_study_data(Path(manifest["data_directory"]))
        if {k: ds.ids[v].tolist() for k, v in split.items()} != manifest["compound_ids"]:
            raise ValueError("Input split differs from the frozen repair plan")
        seed_everything(manifest["seed"], manifest["threads"])
        status(root, "FITTING_TRAIN_ONLY_MOMENTS", n_train=len(split["train"]))
        baseline = fit_baseline(ds.Y[split["train"]], **manifest["fit"])
        baseline.metadata["train_ids"] = ds.ids[split["train"]].tolist()
        baseline.save(root / "closed_form_anchor.npz")
        write_json(root / "anchor_diagnostics.json", baseline.diagnostics)
        old, scaler, config, checkpoint = load_model(manifest["checkpoint"])
        del checkpoint
        config = replace(config, samples=manifest["samples"], mc_chunk_size=manifest["mc_chunk_size"])
        assert list(scaler.train_ids) == manifest["compound_ids"]["train"]
        means = {}
        for part in PARTITIONS:
            ix = split[part]
            predicted = baseline.conditional(ds.Y[ix, :1], [0], [1, 2, 3]).mean
            metrics, traces = score_measurements(ds.Y[ix, 1:], predicted,
                ds.Y[split["train"], 1:].mean(0), scaler.y_scale)
            old_metric = json.loads((Path(manifest["checkpoint"]) / part / "metrics.json").read_text())
            means[part] = dict(n=len(ix), original=old_metric["measurement"]["physical"]["overall"],
                anchored=metrics["physical"]["overall"],
                applies_to=list(ARMS), note="all three new distributions have the same conditional mean")
            np.savez_compressed(root / (part + "_mean_predictions.npz"), ids=ds.ids[ix],
                                prediction_mean=predicted, **traces)
        write_json(root / "mean_results.json", means)
        status(root, "MEAN_RESULTS_COMPLETE", elapsed_seconds=time.monotonic()-started,
               r2={k: v["anchored"]["r2_training_mean"] for k,v in means.items()})
        models = {
            "L_FULL_GAUSSIAN": ClosedFormGaussianAdapter(baseline, scaler),
            "A_MEAN_ANCHOR": MomentRepairModel(old, baseline, scaler, mode="mean_only"),
            "A_MEAN_WITHIN_ANCHOR": MomentRepairModel(old, baseline, scaler, mode="mean_and_within"),
        }
        for part in PARTITIONS:
            for arm in ARMS:
                folder = root / "arms" / arm / part
                status(root, "EVALUATING", arm=arm, partition=part)
                wrapped = GeometryModel(models[arm], scaler)
                arm_config = replace(config, use_library=False) if arm == "L_FULL_GAUSSIAN" else config
                evaluate_partition(wrapped, scaler, ds, split["train"], split[part], arm_config, folder,
                    seed=manifest["seed"], fractions=manifest["fractions"],
                    object_chunk=manifest["object_chunk"], n_bootstrap=manifest["n_bootstrap"],
                    n_random=manifest["n_random"], progress=lambda done,n: status(root, "EVALUATING",
                        arm=arm, partition=part, done=done, n=n,
                        elapsed_seconds=time.monotonic()-started))
                wrapped.save_geometry(folder, ds.Y[split[part]])
                metric = json.loads((folder / "metrics.json").read_text())
                metric["repair"] = dict(arm=arm, neural_gradient_updates=0,
                    anchor_train_ids=manifest["compound_ids"]["train"],
                    covariance_scope=("independent objects; shared signal within each object" if arm == "L_FULL_GAUSSIAN"
                                      else "retained frozen A cross-object environmental sharing"))
                metric["proper_nll"]["checkpoint_score_scope"] = "no new checkpoint selection; anchor fitted on TRAIN only"
                write_json(folder / "metrics.json", metric)
                summarize(root, manifest)
                del wrapped
                gc.collect()
            status(root, "PARTITION_COMPLETE", partition=part,
                   elapsed_seconds=time.monotonic()-started)
        result = summarize(root, manifest)
        status(root, "COMPLETE", complete=result["complete"], elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        status(root, "FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        restore_paused_process(root, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    prep.add_argument("--source-run", required=True)
    prep.add_argument("--paused-pid", type=int)
    run = sub.add_parser("execute")
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.source_run, args.paused_pid)
    else:
        execute(args.output)


if __name__ == "__main__":
    main()
