"""Repaired Gaussian and fixed-target representation diagnostics on opened DEV.

No new JEPA objective, biological relation input, endpoint or certification rule
is introduced by this executable. Historical artifacts remain separate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np


def now():
    return datetime.now(timezone.utc).isoformat()


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(plain(payload), indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n")
    temporary.replace(path)


def event(root, phase, **payload):
    value = dict(utc=now(), phase=phase, **payload)
    with (root / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(plain(value), ensure_ascii=False) + "\n")
    write_json(root / "status.json", value)
    print(json.dumps(plain(value), ensure_ascii=False), flush=True)


def cosine(a, b):
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    if np.any(denominator <= 0):
        raise ValueError("Zero-norm profile: cannot silently drop an experimental unit")
    return np.sum(a * b, axis=-1) / denominator


def actual_gains(Y):
    X, Z1, Z2, V = np.moveaxis(np.asarray(Y), 1, 0)
    baseline = cosine(X, V)
    return np.stack((.5 * (cosine((X + Z1) / 2, V) - baseline) - .01,
                     .5 * (cosine((X + Z2) / 2, V) - baseline) - .01,
                     .5 * (cosine((X + Z1 + Z2) / 3, V) - baseline) - .02), -1)


def prepare(data, output, jepa_run):
    from sklearn.model_selection import StratifiedKFold
    from .objective_comparison import _load_declared_data
    data, output, jepa_run = Path(data).resolve(), Path(output).resolve(), Path(jepa_run).resolve()
    ds, splits, scope = _load_declared_data(data)
    if not np.all(ds.observed_mask) or not np.isfinite(ds.Y).all():
        raise ValueError("The declared complete DEV cohort has changed; do not filter rows")
    output.mkdir(parents=True, exist_ok=False)
    core = splits["train"]
    pool = np.concatenate([splits[k] for k in ("validation", "calibration", "evaluation")])
    # Stratification is based only on initial measurements, never Gamma or V.
    energy = np.log(np.maximum(np.linalg.norm(ds.Y[:, 0], axis=1), 1e-12))
    order = np.lexsort((ds.ids.astype(str), energy))
    strata = np.empty(len(ds), int)
    strata[order] = np.minimum(4, np.arange(len(ds)) * 5 // len(ds))
    seeds = [20260912, 20260913, 20260914]
    folds, probe_folds = [], []
    for repeat, seed in enumerate(seeds):
        splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(ds)), strata)):
            folds.append(dict(repeat=repeat, fold=fold, seed=seed,
                              train_ids=ds.ids[train].tolist(), test_ids=ds.ids[test].tolist()))
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(pool)), strata[pool])):
            probe_folds.append(dict(repeat=repeat, fold=fold, seed=seed,
                train_ids=ds.ids[np.r_[core, pool[train]]].tolist(), test_ids=ds.ids[pool[test]].tolist()))
    config = dict(created_utc=now(), data=str(data), jepa_run=str(jepa_run),
        scope=scope, purpose="OPEN_DEV_BASELINE_AND_REPRESENTATION_DIAGNOSTIC",
        k=200, clip=8., noise_shrinkage=.05, samples=1024, mc_chunk=32,
        object_chunk=8, seed=20260912, n_bootstrap=2000, n_random=2000,
        fractions=[.05, .10, .25], threads=4,
        split_ids={k: ds.ids[ix].tolist() for k, ix in splits.items()},
        folds=folds, probe_folds=probe_folds,
        jepa_arms=["R1_COSINE_3E4", "R2_COSINE_1E4"], jepa_epoch=60,
        biology_kernel_active=False, original_contract_changed=False,
        final_opened=False, fifth_repeat_opened=False,
        new_independent_validation=False,
        python=sys.version)
    write_json(output / "config.json", config)
    project = Path(__file__).resolve().parents[1]
    shutil.copy2(project / "protocols/historical/BASELINE_REPRESENTATION_PLAN.md", output / "PROTOCOL.md")
    shutil.copytree(project / "opal2", output / "source_snapshot" / "opal2",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(project / "tests", output / "source_snapshot" / "tests",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    event(output, "prepared", counts={k: len(v) for k, v in splits.items()})


def load_run(root):
    from .objective_comparison import _load_declared_data
    root = Path(root).resolve()
    cfg = json.loads((root / "config.json").read_text())
    ds, splits, _ = _load_declared_data(Path(cfg["data"]))
    if {k: ds.ids[ix].tolist() for k, ix in splits.items()} != cfg["split_ids"]:
        raise ValueError("Declared DEV identities changed")
    if Path(__file__).resolve().parents[1] != root / "source_snapshot":
        raise ValueError("Execute from this run's source_snapshot")
    return root, cfg, ds, splits


def indices(ds, names):
    lookup = {str(unit): i for i, unit in enumerate(ds.ids)}
    return np.array([lookup[str(unit)] for unit in names], int)


def forecast(model, Y, *, samples, seed, object_chunk=8, mc_chunk=32,
             progress=None):
    """Joint draws shared by three actions; bounded-memory full-space scoring."""
    n, _, d = Y.shape
    mean = np.empty((n, 3)); sd = np.empty_like(mean)
    pnull = np.empty_like(mean); ppos = np.empty_like(mean)
    raw_nll = np.empty(n); mse = np.empty(n); sse = np.empty(n)
    normal_nll = np.empty(n)
    prediction_mean = np.empty((n, 3, d), dtype=np.float32)
    for start in range(0, n, object_chunk):
        end = min(n, start + object_chunk)
        cond = model.conditional(Y[start:end, :1], [0], [1, 2, 3])
        prediction_mean[start:end] = cond.mean
        raw_nll[start:end] = -cond.log_prob(Y[start:end, 1:]) / (3 * d)
        normal_nll[start:end] = raw_nll[start:end] - np.log(model.scale).mean()
        err = (Y[start:end, 1:] - cond.mean) / model.scale
        sse[start:end] = np.square(err).sum((1, 2))
        mse[start:end] = sse[start:end] / (3 * d)
        pieces = []
        for first in range(0, samples, mc_chunk):
            amount = min(mc_chunk, samples - first)
            draw_seed = np.random.SeedSequence([seed, start, first]).generate_state(1)[0]
            draws = cond.sample_joint(amount, int(draw_seed))
            z1, z2, v = draws[:, :, 0], draws[:, :, 1], draws[:, :, 2]
            x = Y[None, start:end, 0]
            c0 = cosine(x, v)
            pieces.append(np.stack((.5 * (cosine((x + z1) / 2, v) - c0) - .01,
                .5 * (cosine((x + z2) / 2, v) - c0) - .01,
                .5 * (cosine((x + z1 + z2) / 3, v) - c0) - .02), -1))
        gains = np.concatenate(pieces, axis=0)
        mean[start:end], sd[start:end] = gains.mean(0), gains.std(0, ddof=1)
        pnull[start:end], ppos[start:end] = (gains <= 0).mean(0), (gains >= .005).mean(0)
        if progress is not None:
            progress(end, n)
    return dict(predicted=mean, predictive_sd=sd, mc_se=sd / np.sqrt(samples),
                p_null=pnull, p_positive=ppos, raw_nll=raw_nll,
                standardized_nll=normal_nll, standardized_mse=mse,
                standardized_sse=sse, prediction_mean=prediction_mean)


def save_forecast(path, ids, Y, values):
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path / "predictions.npz", ids=np.asarray(ids, str),
                        actual=actual_gains(Y), **values)
    actions = ("Z1", "Z2", "Z1Z2")
    fields = ["compound_id"] + [f"{kind}_{a}" for a in actions for kind in
        ("actual", "predicted", "predictive_sd", "mc_se", "p_null", "p_positive")]
    actual = actual_gains(Y)
    with (path / "predictions.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t"); writer.writeheader()
        for i, unit in enumerate(ids):
            row = {"compound_id": str(unit)}
            for j, action in enumerate(actions):
                row[f"actual_{action}"] = actual[i, j]
                for kind in ("predicted", "predictive_sd", "mc_se", "p_null", "p_positive"):
                    row[f"{kind}_{action}"] = values[kind][i, j]
            writer.writerow(row)


def policy_report(root, name, ids, Y, values, train_Y, cfg, seed):
    from .baseline_policy import evaluate_predictions
    report = evaluate_predictions(values["predicted"], values["p_null"], actual_gains(Y), ids,
        fractions=cfg["fractions"], train_actual=actual_gains(train_Y), seed=seed,
        n_bootstrap=cfg["n_bootstrap"], n_random=cfg["n_random"])
    report["measurement_metrics"] = dict(
        raw_nll=float(values["raw_nll"].mean()),
        standardized_nll=float(values["standardized_nll"].mean()),
        standardized_mse=float(values["standardized_mse"].mean()),
        mean_mc_se=values["mc_se"].mean(0).tolist())
    write_json(root / name / "policy.json", report)
    return report


def fixed_baseline(root):
    from .closed_form_baseline import fit_baseline
    root, cfg, ds, splits = load_run(root)
    directory = root / "fixed_baseline"
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir()
    event(root, "fixed_baseline_fit", n=len(splits["train"]))
    model = fit_baseline(ds.Y[splits["train"]], k=cfg["k"], clip=cfg["clip"],
                         noise_shrinkage=cfg["noise_shrinkage"])
    model.save(directory / "model.npz")
    diagnostics = model.diagnostics() if callable(model.diagnostics) else model.diagnostics
    write_json(directory / "model_diagnostics.json", diagnostics)
    for number, split in enumerate(("validation", "calibration", "evaluation")):
        ix = splits[split]
        event(root, "fixed_baseline_predict", split=split, n=len(ix))
        values = forecast(model, ds.Y[ix], samples=cfg["samples"], seed=cfg["seed"] + number,
            object_chunk=cfg["object_chunk"], mc_chunk=cfg["mc_chunk"],
            progress=lambda done, n: event(root, "fixed_baseline_predict", split=split, done=done, n=n))
        save_forecast(directory / split, ds.ids[ix], ds.Y[ix], values)
        policy_report(directory, split, ds.ids[ix], ds.Y[ix], values, ds.Y[splits["train"]], cfg, cfg["seed"] + number)
    event(root, "fixed_baseline_complete")


def stability_baseline(root):
    from .closed_form_baseline import fit_baseline
    root, cfg, ds, splits = load_run(root)
    directory = root / "stability_baseline"
    directory.mkdir(exist_ok=True)
    for f in cfg["folds"]:
        name = f"repeat_{f['repeat']}_fold_{f['fold']}"
        destination = directory / name
        if (destination / "policy.json").exists():
            continue
        if destination.exists():
            raise FileExistsError(f"Incomplete fold must be inspected before resuming: {destination}")
        train, test = indices(ds, f["train_ids"]), indices(ds, f["test_ids"])
        event(root, "stability_baseline_fit", fold=name, train_n=len(train), test_n=len(test))
        model = fit_baseline(ds.Y[train], k=cfg["k"], clip=cfg["clip"], noise_shrinkage=cfg["noise_shrinkage"])
        seed = f["seed"] * 10 + f["fold"]
        values = forecast(model, ds.Y[test], samples=cfg["samples"], seed=seed,
            object_chunk=cfg["object_chunk"], mc_chunk=cfg["mc_chunk"],
            progress=lambda done, n: event(root, "stability_baseline_predict", fold=name, done=done, n=n))
        save_forecast(destination, ds.ids[test], ds.Y[test], values)
        model.save(destination / "model.npz")
        diagnostics = model.diagnostics() if callable(model.diagnostics) else model.diagnostics
        write_json(destination / "model_diagnostics.json", diagnostics)
        policy_report(directory, name, ds.ids[test], ds.Y[test], values, ds.Y[train], cfg, seed)
    event(root, "stability_baseline_complete")


def fixed_features(root, cfg, ds, splits):
    from .closed_form_baseline import ClosedFormBaseline
    from .data import TrainScaler
    from .representation_probe import extract_checkpoint_features
    cache = root / "fixed_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as content:
            if not np.array_equal(content["ids"], ds.ids):
                raise ValueError("Representation cache identities differ")
            return {key: content[key] for key in content.files if key != "ids"}
    model = ClosedFormBaseline.load(root / "fixed_baseline" / "model.npz")
    features = dict(raw=(ds.Y[:, 0] - model.center) / model.scale,
                    reliability=model.reliability_features(ds.Y[:, 0], slot=0))
    expected_scaler = TrainScaler.fit(ds, splits["train"])
    provenance = dict(raw="original affine input", reliability="fixed core383 basis",
                      pretraining_ids=ds.ids[splits["train"]].tolist())
    import torch
    torch.set_num_threads(cfg["threads"])
    for arm in cfg["jepa_arms"]:
        checkpoint = Path(cfg["jepa_run"]) / "arms" / arm / f"epoch_{cfg['jepa_epoch']:03d}.pt"
        event(root, "extract_frozen_jepa", arm=arm)
        values, info = extract_checkpoint_features(ds.Y[:, 0], ds.feature_names,
            cfg["jepa_run"], checkpoint, expected_scaler=expected_scaler)
        if set(info["pretraining_ids"]) != set(ds.ids[splits["train"]]):
            raise ValueError("JEPA training population differs from declared fixed383")
        features[arm] = values
        provenance[arm] = info
    np.savez_compressed(cache, ids=ds.ids, **features)
    write_json(root / "fixed_features_provenance.json", provenance)
    return features


def probe_metrics(Y, prediction, train_mean, scale):
    from .representation_probe import regression_metrics
    scale = np.broadcast_to(scale, np.asarray(train_mean).shape)
    standardized = regression_metrics(Y, prediction, train_mean, scale)
    physical = regression_metrics(Y, prediction, train_mean)
    # Eval-centered R² and training-mean skill are deliberately distinct.
    eval_centered = regression_metrics(Y, prediction, Y.mean(axis=0), scale)
    return dict(standardized=standardized, physical=physical,
                r2_evaluation_mean=eval_centered["overall"]["r2_training_mean"],
                slot_r2_evaluation_mean=[v["r2_training_mean"] for v in eval_centered["slots"]])


def paired_probe_report(metrics, *, seed, n_bootstrap):
    names = list(metrics)
    result = []
    from itertools import combinations
    for a, b in combinations(names, 2):
        av = np.asarray(metrics[a]["standardized"]["per_object_sse"])
        bv = np.asarray(metrics[b]["standardized"]["per_object_sse"])
        delta = bv - av  # Positive means A has lower error than B.
        rng = np.random.default_rng(seed)
        means = delta[rng.integers(0, len(delta), (n_bootstrap, len(delta)))].mean(1)
        result.append(dict(arm_a=a, arm_b=b, b_minus_a_mean_sse=float(delta.mean()),
            relative_to_a_mean_sse=float(delta.mean() / av.mean()),
            percentile95_interval=np.quantile(means, [.025, .975]).tolist(),
            uncertainty_scope="paired compound bootstrap conditional on these fixed batches; exploratory"))
    return result


def run_probes(root, *, stability=False):
    from .closed_form_baseline import ClosedFormBaseline
    from .representation_probe import fit_ridge_probe
    root, cfg, ds, splits = load_run(root)
    features = fixed_features(root, cfg, ds, splits)
    core = splits["train"]
    pool = np.concatenate([splits[k] for k in ("validation", "calibration", "evaluation")])
    scale = ClosedFormBaseline.load(root / "fixed_baseline" / "model.npz").scale
    directory = root / ("stability_probes" if stability else "fixed_probes")
    directory.mkdir(exist_ok=True)
    tasks = cfg["probe_folds"] if stability else [dict(repeat=-1, fold=-1, seed=cfg["seed"],
                train_ids=ds.ids[core].tolist(), test_ids=ds.ids[pool].tolist())]
    for task in tasks:
        name = f"repeat_{task['repeat']}_fold_{task['fold']}" if stability else "fixed383"
        destination = directory / name
        if (destination / "comparisons.json").exists():
            continue
        destination.mkdir(exist_ok=True)
        train, test = indices(ds, task["train_ids"]), indices(ds, task["test_ids"])
        if np.intersect1d(core, test).size:
            raise ValueError("Pretraining compounds may not be probe test objects")
        training_mean = ds.Y[train, 1:].mean(axis=0)
        metrics_by_partition = {"all_test": {}}
        for arm, representation in features.items():
            out = destination / arm
            if (out / "metrics.json").exists():
                report = json.loads((out / "metrics.json").read_text())
            else:
                out.mkdir(exist_ok=True)
                event(root, "probe_fit", fold=name, arm=arm, n_train=len(train), n_test=len(test))
                result = fit_ridge_probe(representation, ds.Y[:, 1:], train, test,
                    alphas=(.001, .01, .1, 1., 10.), inner_splits=3,
                    seed=task["seed"] + max(task["fold"], 0))
                prediction = result["predictions"]
                report = dict(input_dimension=result["input_dimension"],
                    selected_alphas=result["selected_alphas"], alpha_grid=result["alpha_grid"],
                    inner_cv_mse=result["cv_standardized_mse"], inner_folds=result["inner_folds"],
                    representation_fit=result["representation_fit"],
                    train_ids=ds.ids[train].tolist(), test_ids=ds.ids[test].tolist(),
                    target="three future measurements, all unchanged 3617 coordinates",
                    scope="linear readout diagnostic; not information-loss proof",
                    partitions={"all_test": probe_metrics(ds.Y[test, 1:], prediction, training_mean, scale)})
                if not stability:
                    lookup = {int(index): pos for pos, index in enumerate(test)}
                    for split in ("validation", "calibration", "evaluation"):
                        pos = np.array([lookup[int(index)] for index in splits[split]])
                        report["partitions"][split] = probe_metrics(ds.Y[splits[split], 1:],
                            prediction[pos], training_mean, scale)
                np.savez_compressed(out / "predictions.npz", ids=ds.ids[test], predictions=prediction,
                                    train_mean=training_mean, fixed_scale=scale)
                write_json(out / "metrics.json", report)
            for partition, values in report["partitions"].items():
                metrics_by_partition.setdefault(partition, {})[arm] = values
        comparisons = {partition: paired_probe_report(metrics,
            seed=cfg["seed"], n_bootstrap=cfg["n_bootstrap"])
            for partition, metrics in metrics_by_partition.items()}
        write_json(destination / "comparisons.json", comparisons)
        event(root, "probe_fold_complete", fold=name, stability=stability)
    event(root, "stability_probes_complete" if stability else "fixed_probes_complete")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "fixed", "stability", "probes", "probe-stability"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--data")
    parser.add_argument("--jepa-run")
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        if args.phase == "prepare":
            if not args.data or not args.jepa_run:
                parser.error("prepare requires --data and --jepa-run")
            prepare(args.data, args.output, args.jepa_run)
        else:
            try:
                if args.phase == "fixed":
                    fixed_baseline(args.output)
                elif args.phase == "stability":
                    stability_baseline(args.output)
                else:
                    run_probes(args.output, stability=args.phase == "probe-stability")
            except Exception as error:
                event(Path(args.output), "failed", requested_phase=args.phase,
                      error_type=type(error).__name__, error=str(error))
                raise


if __name__ == "__main__":
    main()
