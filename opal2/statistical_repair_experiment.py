"""Bounded full-L mean/dispersion development experiment; no neural training."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import filecmp
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import spearmanr
import torch
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .replicate_diagnostic_experiment import NAMES, WEIGHTS, LEVELS

PARTITIONS = ("validation", "evaluation", "calibration")
ARMS = ("L_REFERENCE", "L_MEAN_RESIDUAL", "L_CONDITIONAL_SCALE")
ALPHAS = (1., 10., 100., 1000.)
SCALES = (.75, 1., 1.25, 1.5, 2., 3., 4.)


def now():
    return datetime.now(timezone.utc).isoformat()


def status(root, state, **kw):
    row = dict(utc=now(), state=state, **kw)
    write_json(root / "status.json", row)
    with (root / "progress.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def prepare(root, reference, diagnostic):
    root, reference, diagnostic = map(lambda p: Path(p).resolve(), (root, reference, diagnostic))
    if root.exists():
        raise FileExistsError("Use a fresh output directory")
    old = json.loads((reference / "run_manifest.json").read_text())
    for path in (reference, diagnostic):
        if json.loads((path / "status.json").read_text())["state"] != "COMPLETE":
            raise ValueError("Both saved reference runs must be complete")
    project = Path(__file__).resolve().parents[1]
    unchanged = {p: filecmp.cmp(project / "opal2" / p,
                  Path(old["source_snapshot"]) / "opal2" / p, shallow=False)
                  for p in ("closed_form_baseline.py", "moment_repair.py", "data.py",
                            "training.py", "biology_kernel_evaluation.py")}
    if not all(unchanged.values()):
        raise ValueError(f"Fixed evaluator or full L changed: {unchanged}")
    n = len(old["compound_ids"]["train"])
    permutation = np.random.default_rng(20260913).permutation(n)
    cut = int(.8*n)
    train_ids = old["compound_ids"]["train"]
    inner = dict(fit=[train_ids[i] for i in permutation[:cut]],
                 check=[train_ids[i] for i in permutation[cut:]])
    root.mkdir(parents=True)
    shutil.copy2(project / "protocols/historical/STATISTICAL_REPAIR_PLAN_20260913.md", root / "PROTOCOL.md")
    snap = root / "source_snapshot"
    for folder in ("opal2", "tests"):
        shutil.copytree(project / folder, snap / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = dict(created_utc=now(), reference_run=str(reference), diagnostic_run=str(diagnostic),
        anchor=str(reference / "closed_form_anchor.npz"), checkpoint=old["checkpoint"],
        data_directory=old["data_directory"], compound_ids=old["compound_ids"],
        inner_ids=inner, inner_split_seed=20260913, source_snapshot=str(snap),
        unchanged=unchanged, arms=list(ARMS), partitions=list(PARTITIONS),
        alphas=list(ALPHAS), scales=list(SCALES), relative_improvement_required=.01,
        fit=old["fit"], samples=2000, mc_chunk_size=32, object_chunk=2, threads=2,
        seed=20260912, fractions=old["fractions"], n_bootstrap=2000, n_random=2000,
        neural_gradient_updates=0, original_endpoint_changed=False,
        original_contract_changed=False, original_split_changed=False,
        qc_exclusions_added=False, final_opened=False, fifth_repeat_opened=False,
        historical_dev=True, formal_certificate=False)
    write_json(root / "run_manifest.json", manifest)
    status(root, "PREPARED")


def closed_contrasts(baseline, y, delta=None, a=1.):
    """Exact physical contrasts of the SAME full L conditional, not PCA-only."""
    conditional = baseline.conditional(y[:, :1], [0], [1, 2, 3])
    mean = conditional.mean if delta is None else conditional.mean + delta
    location = np.einsum("ct,ntd->ncd", WEIGHTS, mean)
    shared = np.square(conditional.shared_factor).sum(-1)
    within = np.square(baseline._noise_factor).sum(-1) + baseline.residual_var
    v = (WEIGHTS.sum(1)[:, None]**2 * shared[None]
         + np.square(WEIGHTS).sum(1)[:, None] * within[None])
    v = v * np.square(baseline.scale)[None] * a
    return location, np.broadcast_to(v, location.shape).copy()


def interval_selection_score(actual, mean, variance):
    """Proper marginal interval scores, equal groups, no coverage-only reward."""
    groups = (slice(0,3), slice(3,6), slice(6,9), slice(9,10))
    residual = np.abs(actual-mean)
    sd = np.sqrt(variance)
    scores = []
    for level in LEVELS:
        half = ndtri((1+level)/2)*sd
        value = 2*half + 2/(1-level)*np.maximum(residual-half, 0)
        scores.append(np.mean([value[:, group].mean() for group in groups]))
    return float(np.mean(scores))


def verify_baseline(root, manifest, baseline, ds, split):
    from .replicate_diagnostics import summarize_contrasts
    out = {}
    for part in PARTITIONS:
        ix = split[part]
        mean, var = closed_contrasts(baseline, ds.Y[ix])
        actual = np.einsum("ct,ntd->ncd", WEIGHTS, ds.Y[ix,1:])
        metrics, _ = summarize_contrasts(actual, mean, var, LEVELS)
        reference = Path(manifest["reference_run"]) / "arms/L_FULL_GAUSSIAN" / part
        with np.load(reference / "predictions.npz", allow_pickle=False) as old:
            if old["ids"].tolist() != ds.ids[ix].tolist():
                raise ValueError("Reference IDs differ")
            maximum = float(np.max(np.abs(old["prediction_mean"]-mean[:,:3])))
            if not np.allclose(old["prediction_mean"], mean[:,:3], rtol=2e-5, atol=2e-5):
                raise ValueError("Full L mean identity failed")
        diag = Path(manifest["diagnostic_run"]) / "arms/L_FULL_GAUSSIAN" / part
        old_metric = json.loads((diag / "metrics.json").read_text())
        cv = np.array([v["coverage"] for v in metrics["contrasts"]])
        old_cv = np.array([v["coverage"] for v in old_metric["contrasts"]])
        covdiff = float(np.max(np.abs(cv-old_cv)))
        if covdiff > 1e-5:
            raise ValueError(f"Analytic contrast coverage identity failed: {part}: {covdiff}")
        with np.load(diag / "per_object.npz", allow_pickle=False) as saved:
            if not np.allclose(var.sum(-1), saved["predicted_variance_sum"], rtol=2e-5, atol=2e-5):
                raise ValueError("Exact contrast variance identity failed")
        out[part] = dict(ids_match=True, max_mean_abs_difference=maximum,
                         max_coverage_difference=covdiff, all_contrast_variances_match=True)
    write_json(root / "baseline_identity.json", dict(passed=True, partitions=out,
        input_clipped=False, slot_mean_subtracted_once=True, full_space=True))


def select_modifications(root, manifest, baseline, ds, split):
    from .closed_form_baseline import fit_baseline
    from .statistical_repair import fit_residual
    lookup = {str(v): i for i,v in enumerate(ds.ids)}
    fit = np.array([lookup[c] for c in manifest["inner_ids"]["fit"]])
    check = np.array([lookup[c] for c in manifest["inner_ids"]["check"]])
    if set(fit) & set(check) or (set(fit) | set(check)) != set(split["train"]):
        raise ValueError("Inner split must partition TRAIN only")
    inner = fit_baseline(ds.Y[fit], **manifest["fit"])
    inner.metadata["train_ids"] = ds.ids[fit].tolist()
    inner.save(root / "inner_fit_baseline.npz")
    location = inner.conditional(ds.Y[check,:1], [0], [1,2,3]).mean
    actual = ds.Y[check,1:]
    mse0 = float(np.square(actual-location).mean())
    candidates = []
    for alpha in manifest["alphas"]:
        correction = fit_residual(inner, ds.Y[fit], ds.ids[fit].tolist(), alpha)
        pred = location + correction.predict(ds.Y[check,0])
        candidates.append(dict(alpha=alpha, physical_mse=float(np.square(actual-pred).mean())))
    best = min(candidates, key=lambda row: row["physical_mse"])
    required = manifest["relative_improvement_required"]
    enabled = best["physical_mse"] <= (1-required)*mse0
    chosen_alpha = best["alpha"]
    residual = None
    if enabled:
        residual = fit_residual(baseline, ds.Y[split["train"]],
                                ds.ids[split["train"]].tolist(), chosen_alpha)
        residual.save(root / "mean_residual.npz")
    # Scale evaluation is separate: internal L mean unchanged, all covariance
    # multiplied, and scores in the same inner affine coordinate system.
    cm, cv = closed_contrasts(inner, ds.Y[check])
    ca = np.einsum("ct,ntd->ncd", WEIGHTS, ds.Y[check,1:])
    center = WEIGHTS.sum(1)[None,:,None] * inner.center[None,None]
    ca, cm = (ca-center)/inner.scale, (cm-center)/inner.scale
    cv = cv / np.square(inner.scale)
    scale_rows = [dict(a=a, interval_score=interval_selection_score(ca,cm,cv*a))
                  for a in manifest["scales"]]
    score0 = next(row["interval_score"] for row in scale_rows if row["a"] == 1.)
    best_scale = min(scale_rows, key=lambda row: (row["interval_score"], abs(row["a"]-1)))
    a = best_scale["a"] if best_scale["interval_score"] <= (1-required)*score0 else 1.
    report = dict(inner_ids=manifest["inner_ids"],
        transforms_fit_only=True, external_partitions_used_for_selection=False,
        mean=dict(baseline_mse=mse0, candidates=candidates, enabled=bool(enabled),
                  chosen_alpha=chosen_alpha, refit_on_full_train=bool(enabled),
                  relative_improvement=1-best["physical_mse"]/mse0),
        scale=dict(candidates=scale_rows, baseline_score=score0, chosen_a=a,
                   relative_improvement=1-best_scale["interval_score"]/score0,
                   meaning="multiplier of full conditional covariance, mean unchanged"),
        criterion="at least 1% internal improvement; otherwise identity baseline")
    write_json(root / "selection.json", report)
    return residual, a


def save_exact(root, arm, part, baseline, ds, ix, residual, a):
    from .replicate_diagnostics import summarize_contrasts
    delta = None if residual is None else residual.predict(ds.Y[ix,0])
    mean, var = closed_contrasts(baseline, ds.Y[ix], delta, a)
    actual = np.einsum("ct,ntd->ncd", WEIGHTS, ds.Y[ix,1:])
    metrics, traces = summarize_contrasts(actual, mean, var, LEVELS)
    for name, row in zip(NAMES, metrics["contrasts"]):
        row["name"] = name
    metrics.update(arm=arm, partition=part, ids=ds.ids[ix].tolist(), formal_certificate=False)
    folder = root / "arms" / arm / part
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "exact_contrasts.json", metrics)
    np.savez_compressed(folder / "exact_contrasts_per_object.npz", ids=ds.ids[ix], **traces)


def summarize(root, manifest):
    result = dict(partitions={}, historical_dev=True, formal_certificate=False,
                  neural_gradient_updates=0, complete=False)
    for part in PARTITIONS:
        rows = {}
        for arm in ARMS:
            folder = root / "arms" / arm / part
            if not (folder / "metrics.json").exists():
                continue
            metric = json.loads((folder / "metrics.json").read_text())
            with np.load(folder / "predictions.npz", allow_pickle=False) as d:
                corr = [float(spearmanr(d["predicted"][:,j],d["actual"][:,j]).statistic)
                        for j in range(3)]
                rows[arm] = dict(n=len(d["ids"]), measurement=metric["measurement"],
                    nll=metric["proper_nll"], predicted_gain=d["predicted"].mean(0).tolist(),
                    actual_gain=d["actual"].mean(0).tolist(), predicted_null=d["p_null"].mean(0).tolist(),
                    actual_null=(d["actual"]<=0).mean(0).tolist(), spearman=corr,
                    utility_crps=metric["utility_crps_by_action"], policy=metric["policy"],
                    exact_contrasts=json.loads((folder / "exact_contrasts.json").read_text()))
        result["partitions"][part] = rows
    result["complete"] = all(len(rows)==len(ARMS) for rows in result["partitions"].values())
    write_json(root / "summary.json", result)
    return result


@torch.no_grad()
def execute(root):
    from .biology_kernel_experiment import _load_study_data
    from .biology_kernel_evaluation import evaluate_partition
    from .closed_form_baseline import ClosedFormBaseline
    from .config import TrainConfig
    from .data import TrainScaler
    from .moment_repair_experiment import GeometryModel
    from .statistical_anomaly_audit import audit_opened_data
    from .statistical_repair import StatisticalRepairAdapter
    from .training import seed_everything
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if str(Path(__file__).resolve().parents[1]) != manifest["source_snapshot"]:
        raise ValueError("Execute recorded source snapshot")
    start = time.monotonic()
    try:
        seed_everything(manifest["seed"], manifest["threads"])
        ds, split, _ = _load_study_data(manifest["data_directory"])
        if ds.Y.shape != (639,4,3617) or {k:ds.ids[ix].tolist() for k,ix in split.items()} != manifest["compound_ids"]:
            raise ValueError("Original opened DEV identities, roles or coordinates changed")
        baseline = ClosedFormBaseline.load(manifest["anchor"])
        scaler = TrainScaler.load(Path(manifest["checkpoint"]) / "scaler.json")
        if list(scaler.train_ids) != manifest["compound_ids"]["train"]:
            raise ValueError("Original scaler TRAIN differs")
        status(root,"VERIFYING_BASELINE")
        verify_baseline(root,manifest,baseline,ds,split)
        status(root,"AUDITING_ANOMALIES")
        write_json(root / "anomaly_audit.json",audit_opened_data(ds,baseline,split))
        status(root,"INNER_TRAIN_SELECTION")
        with threadpool_limits(limits=manifest["threads"]):
            residual,a = select_modifications(root,manifest,baseline,ds,split)
        status(root,"SELECTION_COMPLETE",mean_enabled=residual is not None,a=a,
               elapsed_seconds=time.monotonic()-start)
        config = replace(TrainConfig.load(Path(manifest["checkpoint"])/"config.json"),
                         samples=manifest["samples"],mc_chunk_size=manifest["mc_chunk_size"],
                         use_library=False,threads=manifest["threads"])
        for part in PARTITIONS:
            for arm in ARMS:
                r = residual if arm=="L_MEAN_RESIDUAL" else None
                scale = a if arm=="L_CONDITIONAL_SCALE" else 1.
                folder = root/"arms"/arm/part
                save_exact(root,arm,part,baseline,ds,split[part],r,scale)
                identity = r is None and scale==1.
                if identity:
                    reference = Path(manifest["reference_run"])/"arms/L_FULL_GAUSSIAN"/part
                    for file in reference.iterdir():
                        if file.is_file():
                            shutil.copy2(file,folder/file.name)
                    write_json(folder/"evaluation_origin.json",dict(reused=True,source=str(reference),
                        reason="unaltered full L distribution; no duplicate Monte Carlo"))
                else:
                    model = StatisticalRepairAdapter(baseline,scaler,residual=r,a=scale)
                    wrapped = GeometryModel(model,scaler)
                    status(root,"EVALUATING",arm=arm,partition=part)
                    evaluate_partition(wrapped,scaler,ds,split["train"],split[part],config,folder,
                        seed=manifest["seed"],fractions=manifest["fractions"],
                        object_chunk=manifest["object_chunk"],n_bootstrap=manifest["n_bootstrap"],
                        n_random=manifest["n_random"],progress=lambda done,n: status(root,"EVALUATING",
                            arm=arm,partition=part,done=done,n=n,elapsed_seconds=time.monotonic()-start))
                    wrapped.save_geometry(folder,ds.Y[split[part]])
                    write_json(folder/"evaluation_origin.json",dict(reused=False,source="new declared statistical modification"))
                    del wrapped,model
                metric = json.loads((folder / "metrics.json").read_text())
                metric["statistical_repair"] = dict(
                    arm=arm, mean_residual_enabled=r is not None,
                    full_conditional_covariance_multiplier=scale,
                    reused_reference_outputs=identity,
                    selection_scope="original TRAIN internal 306/77 split only",
                    independent_validation=False)
                # Copied reference metrics retain their source repair provenance;
                # current arm/selection metadata must not imply neural selection.
                metric["checkpoint_score_scope"] = "not applicable: frozen statistical model"
                write_json(folder / "metrics.json", metric)
                summarize(root,manifest)
                gc.collect()
        result=summarize(root,manifest)
        status(root,"COMPLETE",complete=result["complete"],elapsed_seconds=time.monotonic()-start)
    except BaseException as error:
        status(root,"FAILED",error=repr(error),elapsed_seconds=time.monotonic()-start)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","execute"))
    parser.add_argument("--output",required=True)
    parser.add_argument("--reference")
    parser.add_argument("--diagnostic")
    args=parser.parse_args()
    if args.mode=="prepare":
        if not args.reference or not args.diagnostic:
            parser.error("prepare requires --reference and --diagnostic")
        prepare(args.output,args.reference,args.diagnostic)
    else:
        with threadpool_limits(limits=2):
            execute(args.output)


if __name__=="__main__":
    main()
