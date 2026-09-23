"""Frozen first direct-geometry experiment on the already opened four-role DEV."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .biology_kernel_experiment import _load_study_data
from .biology_kernel_evaluation import actual_gains, plain, write_json
from .data import TrainScaler
from .gram_geometry import profiles_to_gram, gram_to_coordinates, coordinates_to_gram, gram_gains
from .gram_model import GramConditionalModel
from .gram_evaluation import (fit_score_scale, evaluate_and_save, paired_score_comparison)
from .objective_analysis import fair_crps


PROJECT = Path(__file__).resolve().parents[1]
PARTITIONS = ("validation", "evaluation", "calibration")
CONFIG = dict(seed=20260914, batch_size=64, learning_rate=.0003, weight_decay=.0001,
    gradient_clip=5., max_epochs=200, min_epochs=40, validation_interval=5,
    patience_checks=8, min_delta=.00001, warmup_steps=30, min_learning_rate=.000003,
    validation_samples=256, samples=2000, threads_per_worker=2, mc_chunk=32,
    object_chunk=2, report_interval=20, n_bootstrap=2000, n_random=2000,
    hidden_dim=64, attention_heads=4, attention_layers=2,
    covariance_shrinkage=.05, log_std_bound=4.)


def now():
    return datetime.now(timezone.utc).isoformat()


def event(root, worker, state, **fields):
    item = plain(dict(utc=now(), worker=worker, state=state, **fields))
    write_json(root / f"{worker}_status.json", item)
    with (root / f"{worker}_progress.jsonl").open("a") as stream:
        stream.write(json.dumps(item, allow_nan=False) + "\n")
    print(json.dumps(item, allow_nan=False), flush=True)
    return item


def _check_ids(ds, split, manifest):
    identities = {k: ds.ids[v].tolist() for k, v in split.items()}
    if identities != manifest["compound_ids"]:
        raise ValueError("The original four DEV partitions changed")
    if ds.Y.shape != (639, 4, 3617):
        raise ValueError("Only the existing complete 639-object, four-role export is allowed")
    return identities


def prepare(data, output, faithful_run, moment_run):
    root, data = Path(output).resolve(), Path(data).resolve()
    faithful, moment = Path(faithful_run).resolve(), Path(moment_run).resolve()
    if root.exists():
        raise FileExistsError("A new Gram experiment directory is required")
    protocol = PROJECT / "protocols/historical/GRAM_PROBABILITY_PLAN_20260914.md"
    source_manifest = json.loads((faithful / "run_manifest.json").read_text())
    ds, split, scope = _load_study_data(data)
    identities = _check_ids(ds, split, source_manifest)
    scalar = TrainScaler.load(faithful / "scaler.json")
    if scalar.train_ids != identities["train"] or scalar.feature_names != ds.feature_names.tolist():
        raise ValueError("Use the existing scaler fitted only on the same TRAIN")
    for part in (faithful, moment):
        prior = json.loads((part / "run_manifest.json").read_text())
        if prior["compound_ids"] != identities or prior.get("final_opened") is not False or prior.get("fifth_repeat_opened") is not False:
            raise ValueError("Reference boundary or identities differ")
    if not (moment / "closed_form_anchor.npz").is_file():
        raise FileNotFoundError("The saved L anchor is required")
    y = torch.as_tensor(ds.Y, dtype=torch.float64)
    grams = profiles_to_gram(y)
    u = gram_to_coordinates(grams)
    reconstructed = coordinates_to_gram(u)
    direct, mapped = actual_gains(ds.Y), gram_gains(grams).numpy()
    if not np.allclose(direct, mapped, atol=1e-12, rtol=1e-10):
        raise ValueError("The original and Gram utilities do not agree")
    if not torch.allclose(grams, reconstructed, atol=1e-9, rtol=1e-9):
        raise ValueError("The fixed geometry coordinates do not reconstruct the cohort")
    train = split["train"]
    raw_u = u.numpy()
    uc, us = raw_u[train].mean(0), raw_u[train].std(0)
    us = np.where(us >= 1e-6, us, 1.)
    lognorm = np.log(np.linalg.norm(ds.Y[:, 0], axis=-1))
    nc, ns = lognorm[train].mean(), lognorm[train].std()
    ns = ns if ns >= 1e-6 else 1.
    root.mkdir(parents=True)
    scalar.save(root / "input_scaler.json")
    shutil.copy2(protocol, root / "PROTOCOL.md")
    snapshot = root / "source_snapshot"
    for folder in ("opal2", "tests"):
        shutil.copytree(PROJECT / folder, snapshot / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(PROJECT / "pyproject.toml", snapshot / "pyproject.toml")
    write_json(root / "preprocessing.json", dict(u_center=uc, u_scale=us,
        lognorm_center=nc, lognorm_scale=ns, score_scale=fit_score_scale(grams.numpy()[train]),
        train_ids=identities["train"], input_coordinate_count=3617, pca_truncation=False,
        endpoint_clipping=False))
    preflight = dict(n=639, roles=4, coordinates=3617,
        max_utility_absolute_error=np.max(np.abs(direct-mapped)),
        max_gram_reconstruction_error=torch.max(torch.abs(grams-reconstructed)).item(),
        min_schur_eigenvalue=torch.linalg.eigvalsh(grams[:, 1:, 1:]-grams[:, 1:, :1]*grams[:, :1, 1:]).min().item(),
        original_endpoint_changed=False, removed_compounds=0,
        role_order=["X", "Z1", "Z2", "V"], new_training_steps=0)
    write_json(root / "geometry_identity_check.json", preflight)
    manifest = dict(created_utc=now(), data_directory=str(data), source_snapshot=str(snapshot),
        faithful_run=str(faithful), moment_run=str(moment), config=CONFIG,
        compound_ids=identities, feature_names=ds.feature_names.tolist(),
        feature_groups=ds.feature_groups, data_shape=list(ds.Y.shape), split_scope=scope,
        historical_dev=True, final_opened=False, fifth_repeat_opened=False,
        original_endpoint_changed=False, original_contract_changed=False,
        original_split_changed=False, formal_certificate=False,
        arms=["L_GRAM", "J10_GRAM", "F10_GRAM", "G_DIRECT"],
        g_input="complete X plus deterministic log norm; no chemistry/reference/JEPA/kernel",
        comparison_scope="information-matched to L; historical J/F have additional inputs",
        selection="minimum validation original ADD_TWO Gamma CRPS, fixed MC stream",
        inference_samples=2000, python_executable=sys.executable)
    write_json(root / "run_manifest.json", manifest)
    event(root, "preparation", "COMPLETE", preflight=preflight,
          split_counts={k: len(v) for k, v in split.items()})
    return manifest


def load_run(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if PROJECT != Path(manifest["source_snapshot"]):
        raise ValueError("Execute the frozen run snapshot")
    if manifest["config"] != CONFIG:
        raise ValueError("The predeclared training configuration changed")
    for key in ("final_opened", "fifth_repeat_opened", "original_endpoint_changed", "original_contract_changed", "original_split_changed"):
        if manifest[key] is not False:
            raise ValueError("The original experiment boundary changed")
    ds, split, _ = _load_study_data(manifest["data_directory"])
    _check_ids(ds, split, manifest)
    if ds.feature_names.tolist() != manifest["feature_names"]:
        raise ValueError("The full feature order changed")
    stats = json.loads((root / "preprocessing.json").read_text())
    if stats["train_ids"] != manifest["compound_ids"]["train"]:
        raise ValueError("Preprocessing training identities differ")
    return root, manifest, ds, split, stats


def _inputs(root, ds, stats):
    scaler = TrainScaler.load(root / "input_scaler.json")
    # Only X is accessed for deployed input. Target measurements are separate.
    x = torch.tensor(scaler.transform_y(ds.Y[:, 0]), dtype=torch.float32)
    lognorm = np.log(np.linalg.norm(ds.Y[:, 0], axis=-1))
    n = torch.tensor((lognorm-stats["lognorm_center"])/stats["lognorm_scale"], dtype=torch.float32).unsqueeze(-1)
    return x, n


@torch.no_grad()
def sample_model(model, x, n, stats, *, samples, seed, object_chunk=32):
    model.eval()
    center = torch.tensor(stats["u_center"], dtype=torch.float64)
    scale = torch.tensor(stats["u_scale"], dtype=torch.float64)
    chunks = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for start in range(0, len(x), object_chunk):
        pred = model(x[start:start+object_chunk], n[start:start+object_chunk])
        u = pred.sample(samples, generator=generator).double()*scale+center
        chunks.append(coordinates_to_gram(u).numpy())
    return np.concatenate(chunks, axis=1)


def _schedule(step, total, cfg):
    if step < cfg["warmup_steps"]:
        return (step+1)/cfg["warmup_steps"]
    fraction = min(1., (step-cfg["warmup_steps"])/max(1, total-cfg["warmup_steps"]))
    floor = cfg["min_learning_rate"]/cfg["learning_rate"]
    return floor+(1-floor)*.5*(1+math.cos(math.pi*fraction))


def train(root):
    root, manifest, ds, split, stats = load_run(root)
    if (root / "training_complete.json").exists():
        return
    cfg = manifest["config"]
    torch.set_num_threads(cfg["threads_per_worker"])
    torch.manual_seed(cfg["seed"])
    x, n = _inputs(root, ds, stats)
    # Training targets and validation selection are kept outside the predictor.
    grams = profiles_to_gram(torch.tensor(ds.Y, dtype=torch.float64))
    u = gram_to_coordinates(grams).numpy()
    target = torch.tensor((u-stats["u_center"])/stats["u_scale"], dtype=torch.float32)
    actual = gram_gains(grams).numpy()
    ix, vi = np.asarray(split["train"]), np.asarray(split["validation"])
    model = GramConditionalModel(manifest["feature_groups"],
        **{k: cfg[k] for k in ("hidden_dim", "attention_heads", "attention_layers", "covariance_shrinkage", "log_std_bound")})
    mean_params, cov_params = list(model.mean_parameters()), list(model.covariance_parameters())
    optimizers = [torch.optim.AdamW(p, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
                  for p in (mean_params, cov_params)]
    steps_per_epoch = math.ceil(len(ix)/cfg["batch_size"])
    schedulers = [torch.optim.lr_scheduler.LambdaLR(o,
        lambda step: _schedule(step, cfg["max_epochs"]*steps_per_epoch, cfg)) for o in optimizers]
    rng = torch.Generator(device="cpu").manual_seed(cfg["seed"]+31)
    started, best, stale, step = time.monotonic(), float("inf"), 0, 0
    best_epoch, last_significant = 0, float("inf")
    folder = root / "arms/G_DIRECT"
    folder.mkdir(parents=True, exist_ok=True)

    def checkpoint(epoch, score):
        return dict(state_dict=deepcopy(model.state_dict()), model_config=model.config,
            epoch=epoch, optimizer_steps=step, validation_gamma_crps=score,
            train_ids=manifest["compound_ids"]["train"],
            validation_ids=manifest["compound_ids"]["validation"],
            feature_names=manifest["feature_names"], selection=manifest["selection"])

    event(root, "training", "STARTED", parameters=sum(p.numel() for p in model.parameters()),
        train=len(ix), validation=len(vi), max_epochs=cfg["max_epochs"],
        device="cpu", chemistry=False, jepa=False, biological_kernel=False)
    history = []
    for epoch in range(cfg["max_epochs"]+1):
        mean_loss, cov_loss, mean_grad, cov_grad = [], [], [], []
        if epoch:
            model.train()
            order = ix[torch.randperm(len(ix), generator=rng).numpy()]
            for start in range(0, len(ix), cfg["batch_size"]):
                batch = order[start:start+cfg["batch_size"]]
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                losses = model.loss(x[batch], n[batch], target[batch])
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("Nonfinite joint-geometry training objective")
                losses["loss"].backward()
                mg = torch.nn.utils.clip_grad_norm_(mean_params, cfg["gradient_clip"], error_if_nonfinite=True)
                cg = torch.nn.utils.clip_grad_norm_(cov_params, cfg["gradient_clip"], error_if_nonfinite=True)
                for opt, scheduler in zip(optimizers, schedulers):
                    opt.step()
                    scheduler.step()
                step += 1
                mean_loss.append(losses["mean_mse"].item())
                cov_loss.append(losses["covariance_nll"].item())
                mean_grad.append(float(mg))
                cov_grad.append(float(cg))
        row = dict(epoch=epoch, optimizer_steps=step, elapsed_seconds=time.monotonic()-started,
            train_u_mean_mse=np.mean(mean_loss) if mean_loss else None,
            train_u_covariance_nll=np.mean(cov_loss) if cov_loss else None,
            mean_gradient_norm=np.mean(mean_grad) if mean_grad else None,
            covariance_gradient_norm=np.mean(cov_grad) if cov_grad else None,
            learning_rate=optimizers[0].param_groups[0]["lr"])
        if epoch % cfg["validation_interval"] == 0:
            samples = sample_model(model, x[vi], n[vi], stats, samples=cfg["validation_samples"],
                                   seed=cfg["seed"]+97001)
            gain_samples = gram_gains(torch.tensor(samples)).numpy()
            score = float(fair_crps(gain_samples, actual[vi]).mean(0)[2])
            with torch.no_grad():
                losses = model.loss(x[vi], n[vi], target[vi])
            row.update(validation_gamma_crps=score,
                validation_u_mean_mse=losses["mean_mse"].item(),
                validation_u_covariance_nll=losses["covariance_nll"].item(),
                validation_predicted_gamma=gain_samples.mean((0, 1)).tolist(),
                validation_actual_gamma=actual[vi].mean(0).tolist(),
                validation_predicted_null=(gain_samples <= 0).mean((0, 1)).tolist())
            if score < best:
                best, best_epoch = score, epoch
                torch.save(checkpoint(epoch, score), folder / "best.pt")
            if score < last_significant-cfg["min_delta"]:
                stale, last_significant = 0, score
            elif epoch:
                stale += 1
            row.update(best_epoch=best_epoch, best_validation_gamma_crps=best,
                       checks_without_significant_improvement=stale)
            event(root, "training", "VALIDATED", **row)
            torch.save({**checkpoint(epoch, score), "optimizers": [o.state_dict() for o in optimizers],
                        "schedulers": [s.state_dict() for s in schedulers]}, folder / "last.pt")
        history.append(plain(row))
        with (folder / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(plain(row), allow_nan=False)+"\n")
        if epoch and epoch % cfg["report_interval"] == 0:
            write_json(folder / f"report_epoch_{epoch:04}.json", dict(history=history[-cfg["report_interval"]:],
                best_epoch=best_epoch, best_validation_gamma_crps=best, no_evaluation_selection=True))
        if epoch >= cfg["min_epochs"] and stale >= cfg["patience_checks"]:
            break
    completion = dict(epoch=epoch, optimizer_steps=step, best_epoch=best_epoch,
        best_validation_gamma_crps=best, elapsed_seconds=time.monotonic()-started,
        stop_reason="validation_patience" if stale >= cfg["patience_checks"] else "epoch_cap",
        final_opened=False, fifth_repeat_opened=False, formal_certificate=False)
    write_json(root / "training_complete.json", completion)
    event(root, "training", "FIT_COMPLETE", **completion)
    payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    for part in PARTITIONS:
        ii = np.asarray(split[part])
        draws = sample_model(model, x[ii], n[ii], stats, samples=cfg["samples"],
                             seed=cfg["seed"]+200000+PARTITIONS.index(part)*1000)
        evaluate_and_save(folder / part, draws, grams.numpy()[ii], ds.ids[ii],
            metadata=dict(arm="G_DIRECT", model_config=model.config, epoch=best_epoch,
                selection=manifest["selection"], input=manifest["g_input"]),
            train_actual_gains=actual[ix], score_scale=stats["score_scale"],
            seed=cfg["seed"], n_bootstrap=cfg["n_bootstrap"], n_random=cfg["n_random"])
        event(root, "training", "PARTITION_EVALUATED", partition=part, best_epoch=best_epoch)
        del draws
    event(root, "training", "COMPLETE", **completion)


def references(root):
    from .gram_reference import ClosedFormGramReference, FrozenFaithfulGramReference
    root, manifest, ds, split, stats = load_run(root)
    cfg = manifest["config"]
    torch.set_num_threads(cfg["threads_per_worker"])
    actual_grams = profiles_to_gram(torch.tensor(ds.Y, dtype=torch.float64)).numpy()
    actual = actual_gains(ds.Y)
    for arm in ("L_GRAM", "J10_GRAM", "F10_GRAM"):
        if arm == "L_GRAM":
            reference = ClosedFormGramReference.from_checkpoint(Path(manifest["moment_run"])/"closed_form_anchor.npz")
        else:
            reference = FrozenFaithfulGramReference.from_run(manifest["faithful_run"],
                arm="J_JOINT" if arm == "J10_GRAM" else "F_MEAN_PROTECTED", expected_epoch=10)
        event(root, "references", "MODEL_LOADED", arm=arm)
        for part in PARTITIONS:
            folder = root / "arms" / arm / part
            if (folder / "metrics.json").exists():
                continue
            ii = np.asarray(split[part])
            event(root, "references", "SAMPLING", arm=arm, partition=part, objects=len(ii))
            inputs = ds.Y[ii, 0] if arm == "L_GRAM" else ii
            result = reference.sample(inputs, n_samples=cfg["samples"],
                seed=cfg["seed"]+200000+PARTITIONS.index(part)*1000,
                object_chunk_size=cfg["object_chunk"], draw_chunk_size=cfg["mc_chunk"])
            # gram_reference returns a typed bundle; no full-spectrum outcomes
            # are passed to the reference's decision-time prediction function.
            draws = result.grams if hasattr(result, "grams") else result["grams"]
            metadata = result.metadata if hasattr(result, "metadata") else result["metadata"]
            evaluate_and_save(folder, draws, actual_grams[ii], ds.ids[ii],
                metadata={**metadata, "arm": arm}, train_actual_gains=actual[split["train"]],
                score_scale=stats["score_scale"], seed=cfg["seed"],
                n_bootstrap=cfg["n_bootstrap"], n_random=cfg["n_random"])
            event(root, "references", "PARTITION_EVALUATED", arm=arm, partition=part)
            del result, draws
        del reference
        gc.collect()
    event(root, "references", "COMPLETE")


def _summarize_unlocked(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    summaries, comparisons = {}, {}
    lines = ["# Joint geometry development comparison", "", "Original four-role DEV, original utilities; no FINAL or fifth repeat.", "",
        "L and G use X information. J/F are historical models with chemistry and references; this is not an isolated output-only ablation.", "",
        "| Partition | Model | ADD_TWO predicted / actual mean | NULL predicted / observed | Gamma CRPS | NULL Brier | Spearman |", "|---|---|---:|---:|---:|---:|---:|"]
    for part in PARTITIONS:
        summaries[part] = {}
        for arm in manifest["arms"]:
            path = root / "arms" / arm / part / "metrics.json"
            if not path.exists():
                continue
            metric = json.loads(path.read_text())
            action = metric["action_metrics"][2]
            summaries[part][arm] = dict(joint_geometry_energy_score=metric["joint_geometry_energy_score"],
                action_metrics=metric["action_metrics"], utility=metric["utility"],
                geometry=metric["geometry"], common_budget=metric["policy"]["common_budget"])
            with np.load(path.parent/"predictions.npz", allow_pickle=False) as stored:
                pn = float(np.mean(stored["p_null"][:, 2]))
            rho = action.get("spearman")
            lines.append(f"| {part} | {arm} | {action['predicted_mean']:.5f} / {action['actual_mean']:.5f} | {pn:.3f} / {action['null_rate']:.3f} | {metric['utility'][2]['crps']:.5f} | {action['null_brier']:.4f} | {rho if rho is not None else 'undefined'} |")
        left, right = (root/"arms"/arm/part/"predictions.npz" for arm in ("G_DIRECT", "L_GRAM"))
        if left.exists() and right.exists():
            comparisons[part] = paired_score_comparison(left, right)
    complete = all(len(items) == 4 for items in summaries.values())
    payload = dict(updated_utc=now(), complete=complete, partitions=summaries,
        paired_G_minus_L=comparisons, historical_dev=True, formal_certificate=False,
        final_opened=False, fifth_repeat_opened=False)
    write_json(root / "summary.json", payload)
    lines += ["", "Joint Gamma samples are scored, not Gamma of a predicted mean Gram.",
        "Paired intervals condition on the fitted rules and existing shared batches. Coverage and selection results are developmental, not certification.",
        "", "Complete: "+str(complete)]
    (root / "REPORT.md").write_text("\n".join(lines)+"\n")
    return payload


def summarize(root):
    root = Path(root).resolve()
    with (root / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _summarize_unlocked(root)


def launch(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    workers = {}
    for name in ("train", "references"):
        command = [manifest["python_executable"], "-u", "-m", "opal2.gram_experiment", name, "--output", str(root)]
        with (root / f"{name}.log").open("xb") as log:
            proc = subprocess.Popen(command, cwd=manifest["source_snapshot"],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        sleep_guard = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(proc.pid)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        workers[name] = dict(pid=proc.pid, caffeinate_pid=sleep_guard.pid, command=command)
    write_json(root / "launch.json", dict(created_utc=now(), workers=workers,
        older_processes_signalled=False, concurrent_workers=2))
    print(json.dumps(workers), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("data", "output", "faithful-run", "moment-run"):
        p.add_argument("--"+name, required=True)
    for name in ("train", "references", "summarize", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.data, args.output, args.faithful_run, args.moment_run)
        return
    if args.command == "launch":
        launch(args.output)
        return
    if args.command == "summarize":
        summarize(args.output)
        return
    try:
        (train if args.command == "train" else references)(args.output)
        summarize(args.output)
    except Exception as error:
        event(Path(args.output).resolve(), "training" if args.command == "train" else "references",
              "FAILED", error=repr(error))
        raise


if __name__ == "__main__":
    main()
