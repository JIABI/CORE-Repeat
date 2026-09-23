"""Exact, no-refit Gaussian replicate checks on the opened four-role DEV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import filecmp
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .biology_kernel_evaluation import write_json


NAMES = ("Z1", "Z2", "V", "Z1_minus_Z2", "Z1_minus_V", "Z2_minus_V",
         "mean_Z1_Z2", "mean_Z1_V", "mean_Z2_V", "mean_Z1_Z2_V")
WEIGHTS = np.asarray([[1,0,0],[0,1,0],[0,0,1],[1,-1,0],[1,0,-1],[0,1,-1],
                      [.5,.5,0],[.5,0,.5],[0,.5,.5],[1/3,1/3,1/3]], float)
ARMS = ("A_ORIGINAL", "L_FULL_GAUSSIAN", "A_MEAN_ANCHOR", "A_MEAN_WITHIN_ANCHOR")
PARTITIONS = ("train", "validation", "evaluation", "calibration")
LEVELS = (.5,.8,.9,.95)


def now():
    return datetime.now(timezone.utc).isoformat()


def update(root, state, **kwargs):
    record = dict(utc=now(), state=state, **kwargs)
    write_json(root / "status.json", record)
    print(json.dumps(record), flush=True)


def prepare(root, repair):
    root, repair = Path(root).resolve(), Path(repair).resolve()
    if root.exists():
        raise FileExistsError("Use a new diagnostic output directory")
    old = json.loads((repair / "run_manifest.json").read_text())
    if json.loads((repair / "status.json").read_text())["state"] != "COMPLETE":
        raise ValueError("The fixed repair reference is not complete")
    project = Path(__file__).resolve().parents[1]
    core = ("model.py", "training.py", "data.py", "closed_form_baseline.py", "moment_repair.py")
    unchanged = {f: filecmp.cmp(project / "opal2" / f,
                              Path(old["source_snapshot"]) / "opal2" / f, shallow=False)
                 for f in core}
    if not all(unchanged.values()):
        raise ValueError(f"Core code differs from the completed repair: {unchanged}")
    root.mkdir(parents=True)
    shutil.copy2(project / "protocols/historical/REPLICATE_DIAGNOSTIC_PLAN_20260913.md", root / "PROTOCOL.md")
    snap = root / "source_snapshot"
    shutil.copytree(project / "opal2", snap / "opal2",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(project / "tests", snap / "tests",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = dict(created_utc=now(), repair_run=str(repair), checkpoint=old["checkpoint"],
        anchor=str(repair / "closed_form_anchor.npz"), data_directory=old["data_directory"],
        compound_ids=old["compound_ids"], arms=list(ARMS), partitions=list(PARTITIONS),
        weights=WEIGHTS.tolist(), contrasts=list(NAMES), levels=list(LEVELS),
        source_snapshot=str(snap), core_unchanged=unchanged, threads=2, object_chunk=2,
        method="exact Gaussian linear contrast marginals in all 3617 coordinates",
        parameter_updates=0, monte_carlo_draws=0, original_contract_changed=False,
        original_endpoint_changed=False, final_opened=False, fifth_repeat_opened=False,
        other_processes_changed=False, historical_dev=True, formal_certificate=False)
    write_json(root / "run_manifest.json", manifest)
    update(root, "PREPARED")


def saved_folder(manifest, arm, part):
    return (Path(manifest["checkpoint"]) / part if arm == "A_ORIGINAL"
            else Path(manifest["repair_run"]) / "arms" / arm / part)


@torch.no_grad()
def execute(root):
    from .biology_kernel_experiment import _load_study_data
    from .closed_form_baseline import ClosedFormBaseline
    from .data import attach_library_context
    from .moment_repair import ClosedFormGaussianAdapter, repair_joint_gaussian
    from .replicate_diagnostics import gaussian_contrast_moments, summarize_contrasts
    from .training import fixed_batch, load_model, seed_everything
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if str(Path(__file__).resolve().parents[1]) != manifest["source_snapshot"]:
        raise ValueError("Run the recorded diagnostic source snapshot")
    start = time.monotonic()
    try:
        ds, split, _ = _load_study_data(manifest["data_directory"])
        if ds.Y.shape != (639,4,3617):
            raise ValueError("Only the declared full four-role DEV is permitted")
        if {k: ds.ids[v].tolist() for k,v in split.items()} != manifest["compound_ids"]:
            raise ValueError("Compound identities or partitions changed")
        seed_everything(20260912, manifest["threads"])
        old, scaler, config, checkpoint = load_model(manifest["checkpoint"])
        baseline = ClosedFormBaseline.load(manifest["anchor"])
        anchor = ClosedFormGaussianAdapter(baseline, scaler)
        if list(scaler.train_ids) != manifest["compound_ids"]["train"]:
            raise ValueError("Scaler fitting population changed")
        normalized = scaler.transform(ds)
        if config.use_library:
            normalized = attach_library_context(normalized, old.library_bank)
        result = dict(created_utc=now(), partitions={}, method=manifest["method"],
                      parameter_updates=0, monte_carlo_draws=0, formal_certificate=False,
                      training_checkpoint_epoch=checkpoint.get("epoch"))
        for part in PARTITIONS:
            ix = split[part]
            n,d = len(ix),ds.Y.shape[-1]
            actual = np.einsum("ct,ntd->ncd", WEIGHTS, ds.Y[ix,1:])
            moments = {arm: (np.empty_like(actual),np.empty_like(actual)) for arm in ARMS}
            nll = {arm: np.empty(n) for arm in ARMS}
            for begin in range(0,n,manifest["object_chunk"]):
                end = min(n,begin+manifest["object_chunk"])
                inputs,target,mask = fixed_batch(normalized,ix[begin:end],config)
                original = old(inputs)
                conditional = anchor._conditional(inputs)
                ldist = anchor(inputs)
                noise = anchor._noise(original.mean)
                variants = {"A_ORIGINAL":original,"L_FULL_GAUSSIAN":ldist,
                    "A_MEAN_ANCHOR":repair_joint_gaussian(original,conditional,
                        latent_rank=old.latent_rank,mode="mean_only"),
                    "A_MEAN_WITHIN_ANCHOR":repair_joint_gaussian(original,conditional,
                        latent_rank=old.latent_rank,mode="mean_and_within",_noise=noise)}
                for arm,distribution in variants.items():
                    m,v = gaussian_contrast_moments(distribution,WEIGHTS,scaler.y_center,scaler.y_scale)
                    moments[arm][0][begin:end],moments[arm][1][begin:end] = m,v
                    logp = distribution.log_prob(target,mask).cpu().numpy()
                    nll[arm][begin:end] = (-logp + 3*np.log(scaler.y_scale).sum())/(3*d)
                del variants,original,ldist,inputs,target,mask,distribution,m,v
                if begin == 0 or end == n or end % 40 == 0:
                    update(root,"EVALUATING",partition=part,done=end,n=n,
                           elapsed_seconds=time.monotonic()-start)
            reports = {}
            for arm,(mean,variance) in moments.items():
                metrics,traces = summarize_contrasts(actual,mean,variance,LEVELS)
                for label,entry in zip(NAMES,metrics["contrasts"]):
                    entry["name"] = label
                metrics.update(arm=arm,partition=part,training_in_sample=(part=="train"),
                    ids=ds.ids[ix].tolist(),raw_nll_mean=float(nll[arm].mean()),
                    raw_nll_median=float(np.median(nll[arm])),raw_nll_max=float(nll[arm].max()),
                    raw_nll_max_id=str(ds.ids[ix[int(nll[arm].argmax())]]),
                    formal_certificate=False)
                # Original saved scores have exactly this population and mean.
                folder=saved_folder(manifest,arm,part)
                if part!="train":
                    with np.load(folder/"predictions.npz",allow_pickle=False) as previous:
                        if previous["ids"].tolist()!=ds.ids[ix].tolist():
                            raise ValueError("Previous saved outputs use different IDs")
                        delta=float(np.max(np.abs(mean[:,:3]-previous["prediction_mean"])))
                        nll_delta=float(np.max(np.abs(nll[arm]-previous["raw_nll"])))
                        if not np.allclose(mean[:,:3],previous["prediction_mean"],rtol=2e-5,atol=2e-5):
                            raise ValueError(f"Fixed prediction changed: {arm}/{part}: {delta}")
                        if not np.allclose(nll[arm],previous["raw_nll"],rtol=2e-5,atol=2e-5):
                            raise ValueError(f"Fixed density changed: {arm}/{part}: {nll_delta}")
                        metrics["previous_output_check"]={"max_mean_abs_difference":delta,
                            "max_nll_abs_difference":nll_delta,
                            "previous_mc_coordinate95":float(previous["coordinate_interval_hits"][:,-1].sum()/(n*3*d)),
                            "previous_mc_difference95":(previous["difference_interval_hits"][:,:,-1].sum(0)/(n*d)).tolist()}
                # Mean/variance geometry diagnostics retain all coordinates, with no fitting.
                future_actual=ds.Y[ix,1:]
                future_mean=mean[:,:3]
                baseline_mean=ds.Y[split["train"],1:].mean(0)
                sse=np.square(future_actual-future_mean).sum((1,2))
                denom=np.square(future_actual-baseline_mean).sum()
                standardized_sse=np.square((future_actual-future_mean)/scaler.y_scale).sum((1,2))
                metrics["mean_accuracy"]={"physical_mse":float(sse.sum()/(n*3*d)),
                    "r2_training_mean":float(1-sse.sum()/denom),
                    "max_standardized_sse_fraction":float(standardized_sse.max()/standardized_sse.sum()),
                    "max_standardized_sse_id":str(ds.ids[ix[int(standardized_sse.argmax())]])}
                out=root/"arms"/arm/part
                out.mkdir(parents=True)
                write_json(out/"metrics.json",metrics)
                np.savez_compressed(out/"per_object.npz",ids=ds.ids[ix],raw_nll=nll[arm],
                    standardized_sse=standardized_sse,physical_sse=sse,**traces)
                reports[arm]=metrics
            result["partitions"][part]=reports
            write_json(root/"summary.json",result)
            del moments,actual,mean,variance
            gc.collect()
        result["complete"]=True
        result["elapsed_seconds"]=time.monotonic()-start
        write_json(root/"summary.json",result)
        update(root,"COMPLETE",complete=True,elapsed_seconds=result["elapsed_seconds"])
    except BaseException as error:
        update(root,"FAILED",error=repr(error),elapsed_seconds=time.monotonic()-start)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","execute"))
    parser.add_argument("--output",required=True)
    parser.add_argument("--repair")
    args=parser.parse_args()
    if args.mode=="prepare":
        if not args.repair:
            parser.error("prepare needs --repair")
        prepare(args.output,args.repair)
    else:
        execute(args.output)


if __name__=="__main__":
    main()
