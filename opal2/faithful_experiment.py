"""Matched J/M/F mean-routing experiment on the already opened four-role DEV.

This first phase compares exact decision-time conditional means. It deliberately
does not report sampled utility, NULL probabilities or a repaired distribution.
The full conditional Gaussian is retained in J/F and in all saved checkpoints;
M is only a mean-learning diagnostic. No old experiment is resumed or modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .biology_kernel_evaluation import plain, score_measurements, write_json
from .biology_kernel_experiment import _load_study_data, configurations
from .config import TrainConfig
from .data import TrainScaler, LibraryBank, fit_library_context, attach_library_context
from .faithful import FaithfulMeasurementModel
from .model import MeasurementWorldModel
from .training import (batches, fixed_batch, model_kwargs, objective_random_stream,
                       random_training_batch, seed_everything,
                       make_world_model_scheduler)


ARMS = ("J_JOINT", "M_MEAN_ONLY", "F_MEAN_PROTECTED")
MODES = dict(zip(ARMS, ("joint", "mean_only", "faithful")))
PARTITIONS = ("train", "validation", "evaluation", "calibration")
SEED = 20260912
EPOCHS = 100
VALIDATION_INTERVAL = 5
REPORT_INTERVAL = 20


def now():
    return datetime.now(timezone.utc).isoformat()


def status(root, state, **fields):
    item = dict(utc=now(), state=state, **plain(fields))
    write_json(root / "status.json", item)
    with (root / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(item, allow_nan=False) + "\n")
    print(json.dumps(item, allow_nan=False), flush=True)
    return item


def configuration():
    base = configurations()[str(SEED)]["A_OFF_GAUSSIAN"]
    return replace(base, epochs=EPOCHS, reference_loss_weight=0.,
                   chemical_regularization_weight=0., patience=EPOCHS + 1,
                   objective="predictive_nll").validate()


def _reference_summary(reference, identities):
    """Copy the existing full-space L scores; do not fit or read future arrays."""
    if reference is None:
        return dict(available=False, reason="No completed reference run supplied", fitted=False)
    reference = Path(reference).resolve()
    manifest = json.loads((reference / "run_manifest.json").read_text())
    if manifest.get("compound_ids") != identities:
        raise ValueError("The L reference must have the identical opened DEV partitions")
    for key in ("final_opened", "fifth_repeat_opened"):
        if manifest.get(key) is not False:
            raise ValueError("The L reference does not preserve the declared data boundary")
    if json.loads((reference / "status.json").read_text()).get("state") != "COMPLETE":
        raise ValueError("The L reference run must already be complete")
    folders = (reference / "arms/L_FULL_GAUSSIAN", reference / "arms/L_REFERENCE")
    folder = next((p for p in folders if p.is_dir()), None)
    if folder is None:
        raise ValueError("No saved full-space L arm exists in the reference run")
    output = dict(available=True, source_run=str(reference), source_arm=str(folder),
                  fitted=False, copied_existing_scores=True, partitions={})
    for part in PARTITIONS:
        path = folder / part / "metrics.json"
        if not path.is_file():
            output["partitions"][part] = dict(available=False)
            continue
        report = json.loads(path.read_text())
        if "measurement" not in report:
            raise ValueError(f"L reference has no matching full-space measurement scores: {path}")
        if int(report.get("n", -1)) != len(identities[part]):
            raise ValueError("L reference partition size differs")
        output["partitions"][part] = dict(available=True, n=report["n"],
            measurement=report["measurement"], source_file=str(path))
    return output


def prepare(data, root, protocol, reference=None):
    """Record the fixed first-phase design and fit TRAIN-only preprocessing."""
    data, root, protocol = (Path(p).resolve() for p in (data, root, protocol))
    if root.exists():
        raise FileExistsError("Use a fresh J/M/F run directory")
    if not protocol.is_file():
        raise FileNotFoundError("Supply the written first-phase protocol")
    project = Path(__file__).resolve().parents[1]
    preflight = project / "FAITHFUL_PREFLIGHT_20260913.json"
    if not preflight.is_file():
        raise FileNotFoundError("The completed full-dimensional preflight report is required")
    dataset, splits, scope = _load_study_data(data)
    config = configuration()
    identities = {k: dataset.ids[v].tolist() for k, v in splits.items()}
    reference_summary = _reference_summary(reference, identities)
    scaler = TrainScaler.fit(dataset, splits["train"], biology_enabled=False)
    normalized = scaler.transform(dataset)
    bank = fit_library_context(normalized, splits["train"]) if config.use_library else None
    root.mkdir(parents=True)
    config.save(root / "config.json")
    scaler.save(root / "scaler.json")
    if bank is not None:
        bank.save(root / "library_context.npz")
    shutil.copy2(protocol, root / "PROTOCOL.md")
    shutil.copy2(preflight, root / preflight.name)
    snapshot = root / "source_snapshot"
    for folder in ("opal2", "tests"):
        shutil.copytree(project / folder, snapshot / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(project / "pyproject.toml", snapshot / "pyproject.toml")
    write_json(root / "L_reference_summary.json", reference_summary)
    manifest = dict(created_utc=now(), purpose="FULL_MEAN_ROUTING_J_M_F_PHASE1_OPEN_DEV",
        data_directory=str(data), data_shape=list(dataset.Y.shape), split_scope=scope,
        compound_ids=identities, feature_names=dataset.feature_names.tolist(),
        config=asdict(config), arms=list(ARMS), arm_modes=MODES,
        source_snapshot=str(snapshot), python_executable=sys.executable,
        seed=SEED, epochs=EPOCHS, steps_per_epoch=math.ceil(len(splits["train"])/config.batch_size),
        optimizer_steps=EPOCHS*math.ceil(len(splits["train"])/config.batch_size),
        validation_interval=VALIDATION_INTERVAL, report_interval=REPORT_INTERVAL,
        primary_checkpoint="fixed final optimizer step; no early stopping",
        auxiliary_checkpoint="lowest validation physical mean MSE at fixed five-epoch checks",
        mean_objective="physical-coordinate MSE on the same randomly drawn legal episodes",
        joint_objective="exact conditional joint predictive NLL; not an ELBO",
        architecture="complete original mean backbone with untied copied covariance decoder",
        same_initial_state=True, interleaved_common_minibatches=True,
        separate_mean_uncertainty_optimizers=True, separate_gradient_clipping=True,
        reference_auxiliary_loss=0., chemical_kl_auxiliary_loss=0.,
        preprocessing_fit_ids=dataset.ids[splits["train"]].tolist(),
        phase1_monte_carlo_draws=0, formal_certificate=False,
        original_endpoint_changed=False, original_contract_changed=False,
        original_split_changed=False, final_opened=False, fifth_repeat_opened=False,
        historical_dev=True, future_utility_repair_claimed=False,
        reference_summary="L_reference_summary.json")
    manifest["preflight_report"] = preflight.name
    write_json(root / "run_manifest.json", manifest)
    status(root, "PREPARED", split_counts={k: len(v) for k, v in splits.items()},
           epochs=EPOCHS, optimizer_steps=manifest["optimizer_steps"],
           training_started=False, preprocessing_fit_on_train_only=True)
    return manifest


def _load_run(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if str(Path(__file__).resolve().parents[1]) != manifest["source_snapshot"]:
        raise ValueError("Execute the recorded source_snapshot, not the live project")
    config = TrainConfig.load(root / "config.json")
    if asdict(config) != manifest["config"] or config.epochs != EPOCHS:
        raise ValueError("The frozen configuration changed")
    if tuple(manifest["arms"]) != ARMS or manifest["arm_modes"] != MODES:
        raise ValueError("The three-arm design changed")
    for key in ("final_opened", "fifth_repeat_opened", "original_endpoint_changed",
                "original_contract_changed", "original_split_changed"):
        if manifest.get(key) is not False:
            raise ValueError("The run does not preserve the declared experiment boundary")
    dataset, splits, _ = _load_study_data(manifest["data_directory"])
    if {k: dataset.ids[v].tolist() for k, v in splits.items()} != manifest["compound_ids"]:
        raise ValueError("The fixed cohort or partitions changed")
    if dataset.feature_names.tolist() != manifest["feature_names"]:
        raise ValueError("The full measurement coordinate order changed")
    scaler = TrainScaler.load(root / "scaler.json")
    if list(scaler.train_ids) != manifest["compound_ids"]["train"]:
        raise ValueError("Preprocessing was not fitted on exactly the recorded TRAIN")
    normalized = scaler.transform(dataset)
    bank = LibraryBank.load(root / "library_context.npz") if config.use_library else None
    if bank is not None:
        normalized = attach_library_context(normalized, bank)
    return root, manifest, dataset, splits, normalized, scaler, bank, config


def _optimizer(parameters, config):
    return torch.optim.AdamW(parameters, lr=config.learning_rate,
                             weight_decay=config.weight_decay)


def _initialize(normalized, scaler, config, bank):
    seed_everything(config.seed, config.threads)
    backbone = MeasurementWorldModel(**model_kwargs(normalized, config))
    backbone.set_outcome_transform(scaler.y_center, scaler.y_scale)
    base = FaithfulMeasurementModel(backbone)
    models = {arm: deepcopy(base) for arm in ARMS}
    if bank is not None:
        for model in models.values():
            model.library_bank = bank
    del base, backbone
    return models


def mean_identity(mean_only, faithful):
    """Compare all original mean-backbone parameters AND buffers exactly."""
    left, right = mean_only.mean_model.state_dict(), faithful.mean_model.state_dict()
    if left.keys() != right.keys():
        raise RuntimeError("M/F mean-backbone state keys differ")
    mismatches, maximum = [], 0.
    for name in left:
        a, b = left[name], right[name]
        if not torch.equal(a, b):
            difference = float((a.to(torch.float64)-b.to(torch.float64)).abs().max())
            maximum = max(maximum, difference)
            mismatches.append(name)
    return dict(exactly_equal=not mismatches, compared_state_tensors=len(left),
                maximum_absolute_difference=maximum, mismatched_tensors=mismatches)


def _assert_identity(root, models, epoch):
    result = dict(epoch=epoch, **mean_identity(models["M_MEAN_ONLY"], models["F_MEAN_PROTECTED"]))
    destination = root / "mean_identity"
    destination.mkdir(exist_ok=True)
    write_json(destination / f"epoch_{epoch:04d}.json", result)
    if not result["exactly_equal"]:
        raise RuntimeError(f"M/F mean trajectory diverged at epoch {epoch}: {result}")
    return result


@torch.no_grad()
def predict_means(model, normalized, indices, scaler, config):
    model.eval()
    output = np.empty((len(indices), 3, len(scaler.y_scale)), dtype=np.float64)
    for start in range(0, len(indices), config.batch_size):
        stop = min(start+config.batch_size, len(indices))
        inputs, _, _ = fixed_batch(normalized, indices[start:stop], config)
        output[start:stop] = scaler.inverse_y(model.exact_predictive_mean(inputs)).cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite fixed-role conditional mean")
    return output


@torch.no_grad()
def validation_mean_metrics(model, dataset, normalized, splits, scaler, config):
    prediction = predict_means(model, normalized, splits["validation"], scaler, config)
    metrics, _ = score_measurements(dataset.Y[splits["validation"], 1:], prediction,
        dataset.Y[splits["train"], 1:].mean(0), scaler.y_scale)
    return dict(n=len(splits["validation"]), measurement=metrics,
                selection="validation physical mean MSE", formal_certificate=False)


def _save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _model_payload(model, config, manifest, epoch, **extra):
    return dict(state_dict=model.state_dict(), model_config=model.config,
        train_config=asdict(config), epoch=epoch,
        optimizer_steps=epoch*manifest["steps_per_epoch"],
        train_ids=manifest["compound_ids"]["train"],
        validation_ids=manifest["compound_ids"]["validation"],
        feature_names=manifest["feature_names"], historical_dev=True,
        formal_certificate=False, **extra)


def _completed_report(path):
    if not path.exists():
        return False
    if not (path / "summary.json").is_file() or not (path / "REPORT.md").is_file():
        raise RuntimeError(f"Incomplete existing report requires inspection: {path}")
    return True


def _number(value):
    return "undefined" if value is None else f"{value:.6g}"


@torch.no_grad()
def mean_report(root, models, dataset, normalized, splits, scaler, config, epoch,
                *, label=None, selected_epochs=None):
    """Evaluate complete fixed-space means; no covariance or utility shortcuts."""
    name = label or f"epoch_{epoch:04d}"
    destination = root / "reports" / name
    if destination.exists():
        raise FileExistsError(f"A report already exists: {destination}")
    folder = destination.with_name(f".{name}.in_progress_{os.getpid()}_{time.time_ns()}")
    folder.mkdir(parents=True)
    train_mean = dataset.Y[splits["train"], 1:].mean(0)
    result = dict(created_utc=now(), epoch=epoch, optimizer_steps=epoch*math.ceil(len(splits["train"])/config.batch_size),
        checkpoint_kind="auxiliary_validation_selected" if selected_epochs else "fixed_step",
        selected_epochs=selected_epochs, original_coordinate_count=dataset.Y.shape[-1],
        historical_dev=True, formal_certificate=False, monte_carlo_draws=0,
        distribution_repair_evaluated=False, utility_or_NULL_repair_claimed=False,
        M_is_mean_diagnostic_not_a_final_probabilistic_model=True, arms={})
    for arm, model in models.items():
        model.eval()
        result["arms"][arm] = {}
        for part in PARTITIONS:
            ix = splits[part]
            prediction = predict_means(model, normalized, ix, scaler, config)
            actual = dataset.Y[ix, 1:]
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite full-space mean prediction")
            scores, traces = score_measurements(actual, prediction, train_mean, scaler.y_scale)
            target = folder / arm / part
            target.mkdir(parents=True)
            np.savez_compressed(target / "predictions.npz", ids=dataset.ids[ix],
                feature_names=dataset.feature_names, prediction_mean=prediction,
                actual_future=actual, **traces)
            metrics = dict(n=len(ix), measurement=scores, checkpoint_kind=result["checkpoint_kind"],
                           future_utility_evaluated=False, formal_certificate=False)
            write_json(target / "metrics.json", metrics)
            result["arms"][arm][part] = metrics
        gc.collect()
    result["L_reference"] = json.loads((root / "L_reference_summary.json").read_text())
    write_json(folder / "summary.json", result)
    lines = [f"# J/M/F mean-routing report: {name}", "",
             "Full 3,617-coordinate conditional means; the original four roles and DEV splits are unchanged.",
             "This is a mean-learning diagnosis, not evidence that the joint distribution, Γ or NULL probabilities have been repaired.", "",
             "| Arm | Partition | Physical MSE | Physical R² versus TRAIN mean | Standardized MSE |", "|---|---|---:|---:|---:|"]
    for arm, parts in result["arms"].items():
        for part, metrics in parts.items():
            score = metrics["measurement"]
            raw, std = score["physical"]["overall"], score["standardized"]["overall"]
            lines.append(f"| {arm} | {part} | {_number(raw['mse'])} | {_number(raw['r2_training_mean'])} | {_number(std['mse'])} |")
    (folder / "REPORT.md").write_text("\n".join(lines)+"\n")
    folder.replace(destination)
    return result


def execute(root, *, resume=False):
    root, manifest, dataset, splits, normalized, scaler, bank, config = _load_run(root)
    if (root / "completion.json").exists():
        raise FileExistsError("The fixed 100-epoch experiment is already complete")
    if (root / "last.pt").exists() != resume:
        raise ValueError("Explicit --resume is required exactly when last.pt exists")
    models = _initialize(normalized, scaler, config, bank)
    params = {arm: dict(mean=list(model.mean_parameters()),
                        uncertainty=list(model.uncertainty_parameters())) for arm, model in models.items()}
    optimizers, schedulers = {}, {}
    for arm in ARMS:
        mean, covariance = params[arm]["mean"], params[arm]["uncertainty"]
        if set(map(id, mean)) & set(map(id, covariance)):
            raise RuntimeError("Mean and covariance parameter groups overlap")
        groups = ("mean",) if arm == "M_MEAN_ONLY" else ("mean", "uncertainty")
        optimizers[arm] = {g: _optimizer(params[arm][g], config) for g in groups}
        schedulers[arm] = {g: make_world_model_scheduler(optimizers[arm][g], config, manifest["steps_per_epoch"])
                           for g in groups}
    rng = np.random.default_rng(config.seed+29)
    start_epoch, prior_elapsed, best = 0, 0., {arm: dict(mse=None, epoch=None) for arm in ARMS}
    if resume:
        checkpoint = torch.load(root / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != asdict(config) or checkpoint["compound_ids"] != manifest["compound_ids"]:
            raise ValueError("Resume configuration or cohort differs")
        for arm in ARMS:
            models[arm].load_state_dict(checkpoint["models"][arm], strict=True)
            for group in optimizers[arm]:
                optimizers[arm][group].load_state_dict(checkpoint["optimizers"][arm][group])
                schedulers[arm][group].load_state_dict(checkpoint["schedulers"][arm][group])
        rng.bit_generator.state = checkpoint["episode_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        start_epoch, prior_elapsed, best = checkpoint["epoch"], checkpoint["elapsed_seconds"], checkpoint["best"]
    else:
        _save_torch(root / "initial_state.pt", _model_payload(models[ARMS[0]], config, manifest, 0))
        initial = models[ARMS[0]].state_dict()
        for arm in ARMS[1:]:
            other = models[arm].state_dict()
            if initial.keys() != other.keys() or any(not torch.equal(initial[k], other[k]) for k in initial):
                raise RuntimeError("J/M/F initial states are not identical")
        write_json(root / "initialization_comparison.json", dict(all_state_tensors_exactly_equal=True,
            arms=list(ARMS), parameter_counts={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},
            mean_parameter_count=sum(p.numel() for p in params[ARMS[0]]["mean"]),
            uncertainty_parameter_count=sum(p.numel() for p in params[ARMS[0]]["uncertainty"])))
    _assert_identity(root, models, start_epoch)
    (root / "root.pid").write_text(str(os.getpid())+"\n")
    started = time.monotonic()
    try:
        if resume and start_epoch and start_epoch % REPORT_INTERVAL == 0:
            if not _completed_report(root / "reports" / f"epoch_{start_epoch:04d}"):
                mean_report(root, models, dataset, normalized, splits, scaler, config, start_epoch)
        status(root, "TRAINING", pid=os.getpid(), resume=resume, next_epoch=start_epoch+1,
               total_epochs=EPOCHS, early_stopping=False)
        for epoch in range(start_epoch, EPOCHS):
            epoch_start = time.monotonic()
            for model in models.values():
                model.train()
            totals = {a: dict(loss=0., mean_mse=0., nll=0., compounds=0, observed_coordinates=0,
                             mean_gradient_norm=0., uncertainty_gradient_norm=0.,
                             mean_clipped_steps=0, uncertainty_clipped_steps=0, steps=0)
                      for a in ARMS}
            for minibatch, ix in enumerate(batches(rng.permutation(splits["train"]), config.batch_size)):
                batch_start = time.monotonic()
                first_step_seconds = {}
                item = random_training_batch(normalized, ix, rng, config)
                for arm in ARMS:
                    arm_start = time.monotonic()
                    for optimizer in optimizers[arm].values():
                        optimizer.zero_grad(set_to_none=True)
                    with objective_random_stream(config, epoch, minibatch):
                        result = models[arm].loss(item["inputs"], item["target_y"],
                                                 item["target_mask"], mode=MODES[arm])
                        loss = result["loss"]
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Nonfinite {arm} loss")
                        loss.backward()
                    record = totals[arm]
                    observed_coordinates = int(result["observed_coordinates"])
                    for key in ("loss", "mean_mse", "nll"):
                        if result.get(key) is not None:
                            record[key] += float(result[key].detach())*observed_coordinates
                    record["compounds"] += len(ix)
                    record["observed_coordinates"] += observed_coordinates
                    for group, optimizer in optimizers[arm].items():
                        norm = torch.nn.utils.clip_grad_norm_(params[arm][group], config.gradient_clip,
                                                             error_if_nonfinite=True)
                        record[f"{group}_gradient_norm"] += float(norm)
                        record[f"{group}_clipped_steps"] += int(float(norm)>config.gradient_clip)
                        optimizer.step()
                        schedulers[arm][group].step()
                    record["steps"] += 1
                    del loss, result
                    if epoch == start_epoch and minibatch == 0:
                        first_step_seconds[arm] = time.monotonic()-arm_start
                del item
                if epoch == start_epoch and minibatch == 0:
                    identity = mean_identity(models["M_MEAN_ONLY"], models["F_MEAN_PROTECTED"])
                    write_json(root / f"first_minibatch_identity_after_epoch_{start_epoch:04d}.json",
                        dict(epoch=epoch+1, minibatch=1, **identity))
                    if not identity["exactly_equal"]:
                        raise RuntimeError(f"M/F mean state diverged on first matched minibatch: {identity}")
                    batch_seconds = time.monotonic()-batch_start
                    status(root, "FIRST_MINIBATCH_COMPLETE", epoch=epoch+1, minibatch=1,
                        optimizer_steps=epoch*manifest["steps_per_epoch"]+1,
                        arm_step_seconds=first_step_seconds, batch_elapsed_seconds=batch_seconds,
                        mean_identity=identity,
                        estimated_remaining_training_seconds=batch_seconds*(manifest["optimizer_steps"]-epoch*manifest["steps_per_epoch"]-1),
                        estimate_excludes_validation_reports_checkpoint_io=True)
            completed = epoch+1
            validation = {}
            validation_records = {}
            if completed % VALIDATION_INTERVAL == 0:
                for arm in ARMS:
                    validation_records[arm] = validation_mean_metrics(models[arm], dataset, normalized, splits, scaler, config)
                    value = validation_records[arm]["measurement"]["physical"]["overall"]["mse"]
                    validation[arm] = value
                    if best[arm]["mse"] is None or value < best[arm]["mse"]:
                        best[arm] = dict(mse=value, epoch=completed)
                        target = root / "arms" / arm
                        target.mkdir(parents=True, exist_ok=True)
                        _save_torch(target / "best_validation_mean.pt", _model_payload(models[arm], config,
                            manifest, completed, selection="validation physical mean MSE", validation_mse=value))
                validation_folder = root / "validation"
                validation_folder.mkdir(exist_ok=True)
                write_json(validation_folder / f"epoch_{completed:04d}.json", dict(epoch=completed,
                    optimizer_steps=completed*manifest["steps_per_epoch"], arms=validation_records,
                    best_validation_mean=best, fixed_role=True, training_continues=completed<EPOCHS,
                    formal_certificate=False, distribution_repair_evaluated=False))
            elapsed = prior_elapsed+time.monotonic()-started
            mean_identity_result = None
            if completed == 1 or completed % REPORT_INTERVAL == 0:
                mean_identity_result = _assert_identity(root, models, completed)
            checkpoint = dict(config=asdict(config), compound_ids=manifest["compound_ids"],
                model_configs={a:m.config for a,m in models.items()}, models={a:m.state_dict() for a,m in models.items()},
                optimizers={a:{g:o.state_dict() for g,o in groups.items()} for a,groups in optimizers.items()},
                schedulers={a:{g:s.state_dict() for g,s in groups.items()} for a,groups in schedulers.items()},
                epoch=completed, optimizer_steps=completed*manifest["steps_per_epoch"],
                best=best, episode_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), elapsed_seconds=elapsed)
            _save_torch(root / "last.pt", checkpoint)
            del checkpoint
            records = {}
            for arm, values in totals.items():
                steps, count = values["steps"], values["observed_coordinates"]
                records[arm] = dict(training_loss=values["loss"]/count,
                    training_compounds=values["compounds"], observed_target_coordinates=count,
                    training_physical_mean_mse=values["mean_mse"]/count,
                    training_joint_nll=(None if arm=="M_MEAN_ONLY" else values["nll"]/count),
                    mean_gradient_norm_before_clip=values["mean_gradient_norm"]/steps,
                    mean_clipped_step_fraction=values["mean_clipped_steps"]/steps,
                    uncertainty_gradient_norm_before_clip=(None if arm=="M_MEAN_ONLY" else values["uncertainty_gradient_norm"]/steps),
                    uncertainty_clipped_step_fraction=(None if arm=="M_MEAN_ONLY" else values["uncertainty_clipped_steps"]/steps),
                    learning_rate=optimizers[arm]["mean"].param_groups[0]["lr"],
                    validation_physical_mean_mse=validation.get(arm), best_validation_mean=best[arm])
                if arm in validation_records:
                    records[arm]["validation_measurement"] = validation_records[arm]["measurement"]
            status(root, "EPOCH_COMPLETE", pid=os.getpid(), epoch=completed, optimizer_steps=completed*manifest["steps_per_epoch"],
                maximum_optimizer_steps=manifest["optimizer_steps"], arms=records,
                epoch_elapsed_seconds=time.monotonic()-epoch_start, elapsed_seconds=elapsed,
                estimated_remaining_seconds=(elapsed/completed)*(EPOCHS-completed),
                estimated_first20_remaining_seconds=(elapsed/completed)*max(REPORT_INTERVAL-completed, 0),
                eta_basis="observed elapsed runtime; machine sleep/contention can increase wall-clock time",
                mean_identity=mean_identity_result, early_stopping=False)
            if completed % REPORT_INTERVAL == 0:
                report_path = root / "reports" / f"epoch_{completed:04d}"
                if not _completed_report(report_path):
                    status(root, "REPORTING_MEANS", epoch=completed, optimizer_steps=completed*manifest["steps_per_epoch"])
                    mean_report(root, models, dataset, normalized, splits, scaler, config, completed)
                status(root, "MEAN_REPORT_READY", epoch=completed,
                    report=str(report_path / "REPORT.md"), summary=str(report_path / "summary.json"),
                    training_continues=completed<EPOCHS, distribution_repair_evaluated=False)
        for arm, model in models.items():
            target = root / "arms" / arm
            target.mkdir(parents=True, exist_ok=True)
            _save_torch(target / "final.pt", _model_payload(model, config, manifest, EPOCHS,
                selection="fixed final step", training_mode=MODES[arm]))
        final_identity = _assert_identity(root, models, EPOCHS)
        # Auxiliary best-mean checkpoints are never substituted for the fixed-step report.
        selected_epochs = {}
        for arm in ARMS:
            payload = torch.load(root / "arms" / arm / "best_validation_mean.pt", map_location="cpu", weights_only=True)
            models[arm].load_state_dict(payload["state_dict"], strict=True)
            selected_epochs[arm] = payload["epoch"]
            del payload
        if not _completed_report(root / "reports/auxiliary_best_validation_mean"):
            mean_report(root, models, dataset, normalized, splits, scaler, config, EPOCHS,
                        label="auxiliary_best_validation_mean", selected_epochs=selected_epochs)
        completion = dict(utc=now(), epochs=EPOCHS, optimizer_steps=manifest["optimizer_steps"],
            arms=list(ARMS), final_mean_identity=final_identity, final_opened=False, fifth_repeat_opened=False,
            original_endpoint_changed=False, original_contract_changed=False, historical_dev=True,
            formal_certificate=False, distribution_repair_evaluated=False,
            elapsed_seconds=prior_elapsed+time.monotonic()-started,
            primary_report=str(root / "reports/epoch_0100/REPORT.md"),
            auxiliary_report=str(root / "reports/auxiliary_best_validation_mean/REPORT.md"))
        write_json(root / "completion.json", completion)
        status(root, "COMPLETE", **{k:v for k,v in completion.items() if k!="utc"})
        return completion
    except BaseException as error:
        status(root, "FAILED", error_type=type(error).__name__, error=str(error),
               resume_available=(root / "last.pt").is_file(), pid=os.getpid())
        raise


def launch(root, *, resume=False):
    """Launch only this new run, detached; never alter an existing worker."""
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if (root / "completion.json").exists():
        raise FileExistsError("This run has already completed")
    pid_path = root / "root.pid"
    if pid_path.exists():
        pid = int(pid_path.read_text().strip())
        check = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
        if check.returncode == 0 and "opal2.faithful_experiment" in check.stdout and str(root) in check.stdout:
            raise RuntimeError(f"The recorded run is already active (PID {pid})")
    if (root / "last.pt").exists() != resume:
        raise ValueError("Use --resume exactly when this run contains last.pt")
    command = [manifest["python_executable"], "-u", "-m", "opal2.faithful_experiment",
               "execute", "--output", str(root)]
    if resume:
        command.append("--resume")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    status(root, "LAUNCHING", resume=resume)
    with (root / "worker.log").open("ab") as output:
        process = subprocess.Popen(command, cwd=manifest["source_snapshot"], env=env,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    pid_path.write_text(str(process.pid)+"\n")
    inhibitor = None
    if sys.platform == "darwin" and Path("/usr/bin/caffeinate").is_file():
        inhibitor = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(process.pid)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True).pid
    result = dict(utc=now(), pid=process.pid, command=command, cwd=manifest["source_snapshot"],
                  worker_log=str(root / "worker.log"), resume=resume, caffeinate_pid=inhibitor,
                  display_may_sleep=True, system_sleep_not_guaranteed_when_lid_closed=True)
    write_json(root / "launch.json", result)
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--protocol", required=True)
    prep.add_argument("--reference-run")
    for name in ("execute", "launch"):
        command = sub.add_parser(name)
        command.add_argument("--output", required=True)
        command.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.data, args.output, args.protocol, args.reference_run)
    elif args.command == "execute":
        execute(args.output, resume=args.resume)
    else:
        launch(args.output, resume=args.resume)


if __name__ == "__main__":
    main()
