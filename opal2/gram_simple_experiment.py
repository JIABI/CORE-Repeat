"""Two complete, same-target statistical geometry baselines on the opened DEV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .biology_kernel_experiment import _load_study_data
from .biology_kernel_evaluation import write_json
from .data import TrainScaler
from .gram_geometry import profiles_to_gram, gram_to_coordinates, coordinates_to_gram, gram_gains
from .gram_evaluation import evaluate_and_save, paired_score_comparison


PROJECT = Path(__file__).resolve().parents[1]
PARTITIONS = ("validation", "evaluation", "calibration")
NEW_ARMS = ("GLOBAL_GEOMETRY", "RIDGE_GEOMETRY")
REFERENCES = ("L_GRAM", "G_DIRECT")
CONFIG = dict(seed=20260914, samples=2000, n_bootstrap=2000, n_random=2000,
              threads=2, ridge_lambdas=[.01, .1, 1., 10., 100.], outer_folds=5,
              inner_folds=4, selection_folds=5)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _event(root, state, **kwargs):
    item = dict(utc=_now(), state=state, **kwargs)
    write_json(root / "status.json", item)
    print(json.dumps(item, allow_nan=False), flush=True)


def prepare(reference, output):
    reference, root = Path(reference).resolve(), Path(output).resolve()
    if (root / "run_manifest.json").exists():
        raise FileExistsError("Do not overwrite a prepared run")
    if root.exists() and any(p.name != "neighbor_support" for p in root.iterdir()):
        raise FileExistsError("New output may contain only the parallel neighbor audit")
    prior = json.loads((reference / "run_manifest.json").read_text())
    for key in ("final_opened", "fifth_repeat_opened", "original_endpoint_changed",
                "original_contract_changed", "original_split_changed"):
        if prior[key] is not False:
            raise ValueError("Original scope changed: " + key)
    if not json.loads((reference / "summary.json").read_text())["complete"]:
        raise ValueError("The reference experiment must be complete")
    ds, split, scope = _load_study_data(prior["data_directory"])
    ids = {k: ds.ids[v].tolist() for k, v in split.items()}
    if ids != prior["compound_ids"] or ds.Y.shape != (639, 4, 3617):
        raise ValueError("Original four-role DEV identities changed")
    for arm in REFERENCES:
        for part in PARTITIONS:
            if not (reference / "arms" / arm / part / "predictions.npz").is_file():
                raise FileNotFoundError(f"Missing saved reference {arm}/{part}")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("input_scaler.json", "preprocessing.json"):
        shutil.copy2(reference / name, root / name)
    shutil.copy2(PROJECT / "protocols/historical/GRAM_SIMPLE_PLAN_20260914.md", root / "PROTOCOL.md")
    snapshot = root / "source_snapshot"
    for name in ("opal2", "tests"):
        shutil.copytree(PROJECT / name, snapshot / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(PROJECT / "pyproject.toml", snapshot / "pyproject.toml")
    manifest = dict(created_utc=_now(), source_snapshot=str(snapshot),
                    data_directory=prior["data_directory"], reference_run=str(reference),
                    compound_ids=ids, config=CONFIG, data_shape=list(ds.Y.shape),
                    split_scope=scope, arms=list(NEW_ARMS), references=list(REFERENCES),
                    final_opened=False, fifth_repeat_opened=False, original_endpoint_changed=False,
                    original_contract_changed=False, original_split_changed=False,
                    historical_dev=True, formal_certificate=False, neural_training=False,
                    penalty_selection="TRAIN only; no held-out model selection")
    write_json(root / "run_manifest.json", manifest)
    _event(root, "PREPARED", n=639, split_counts={k:len(v) for k,v in split.items()})


def load_run(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if PROJECT != Path(manifest["source_snapshot"]) or manifest["config"] != CONFIG:
        raise ValueError("Use the frozen experiment source and configuration")
    for k in ("final_opened", "fifth_repeat_opened", "original_endpoint_changed",
              "original_contract_changed", "original_split_changed"):
        if manifest[k] is not False:
            raise ValueError("Experiment scope changed")
    ds, split, _ = _load_study_data(manifest["data_directory"])
    if {k:ds.ids[v].tolist() for k,v in split.items()} != manifest["compound_ids"]:
        raise ValueError("Compound identities changed")
    stats = json.loads((root / "preprocessing.json").read_text())
    scaler = TrainScaler.load(root / "input_scaler.json")
    if scaler.train_ids != manifest["compound_ids"]["train"] or scaler.feature_names != ds.feature_names.tolist():
        raise ValueError("Original G preprocessing changed")
    # Same input precision/transform as G's deployed input, then float64 algebra.
    xx = scaler.transform_y(ds.Y[:, 0]).astype(np.float32).astype(np.float64)
    lognorm = np.log(np.linalg.norm(ds.Y[:, 0], axis=-1))
    nn = ((lognorm-stats["lognorm_center"])/stats["lognorm_scale"]).astype(np.float32).astype(np.float64)
    x = np.column_stack((xx, nn))
    actual_g = profiles_to_gram(torch.tensor(ds.Y, dtype=torch.float64)).numpy()
    u = gram_to_coordinates(torch.tensor(actual_g)).numpy()
    u = (u-np.asarray(stats["u_center"]))/np.asarray(stats["u_scale"])
    return root, manifest, ds, split, stats, x, u, actual_g


def coordinate_scores(actual, mean, train_mean):
    residual = np.asarray(actual)-mean
    sse = np.square(residual).sum()
    denominator = np.square(actual-train_mean).sum()
    return dict(mse=float(np.square(residual).mean()),
                r2_vs_train_mean=float(1-sse/denominator) if denominator > 0 else None,
                per_coordinate_mse=np.square(residual).mean(0).tolist(),
                mean_residual=residual.mean(0).tolist())


def _mask(row, ids, global_model):
    if global_model:
        return np.full(len(ids), row["selected_n"]/len(ids))
    selected = set(row["selected_ids"])
    return np.array([str(i) in selected for i in ids], dtype=float)


def paired_policy(left, right, left_trace, right_trace, *, left_global=False,
                  right_global=False, seed=20260914, n_bootstrap=2000):
    """Fixed masks; constant-law arms use uniform subset expectations."""
    with np.load(left_trace, allow_pickle=False) as a, np.load(right_trace, allow_pickle=False) as b:
        ids, actual = a["ids"], a["actual"]
        if not np.array_equal(ids, b["ids"]) or not np.array_equal(actual, b["actual"]):
            raise ValueError("Paired policies require identical ordered objects/outcomes")
    rng = np.random.default_rng(seed)
    boot = rng.integers(len(ids), size=(n_bootstrap,len(ids)))
    rows = []
    actions = ("Z1", "Z2", "Z1Z2")
    for section in ("within_action", "common_budget"):
        lookup = {r["label"]:r for r in right["policy"][section]}
        for lrow in left["policy"][section]:
            rrow = lookup[lrow["label"]]
            for key in ("selected_n", "used_wells", "budget_wells", "action", "ranking"):
                if lrow[key] != rrow[key]:
                    raise ValueError("Different policy budget/action in a pair")
            n, k = len(ids), lrow["selected_n"]
            q = k/n
            j = actions.index(lrow["action"])
            delta = _mask(lrow,ids,left_global)-_mask(rrow,ids,right_global)
            value = delta*actual[:,j]
            null = (actual[:,j] <= 0).astype(float)
            false = delta*null
            def estimate(values, denominator=1.):
                if not denominator:
                    return None
                return dict(mean=float(values.mean()/denominator),
                            interval95=np.quantile(values[boot].mean(1)/denominator,[.025,.975]).tolist())
            rows.append(dict(section=section, label=lrow["label"], action=lrow["action"],
                ranking=lrow["ranking"], fraction=lrow["fraction"], selected_n=k,
                used_wells=lrow["used_wells"], value_per_eligible=estimate(value),
                value_per_selected=estimate(value,q), fdp=estimate(false,q),
                fpr=estimate(false,float(null.mean())),
                selected_overlap=None if left_global or right_global else int(np.sum(
                    _mask(lrow,ids,False)*_mask(rrow,ids,False)))))
    return dict(rows=rows, direction="left minus right", formal_certificate=False,
        global_ties="uniform subset expectation, not lexical-ID selected outcomes",
        interval_scope="compound bootstrap conditional on fitted masks/shared batches; original full-partition denominators fixed",
        no_new_holdout=True)


def execute(root):
    from .gram_simple_models import fit_global, fit_ridge, GramSimpleGaussian
    started = time.monotonic()
    root, manifest, ds, split, stats, x, u, grams = load_run(root)
    torch.set_num_threads(CONFIG["threads"])
    train = np.asarray(split["train"])
    reference = Path(manifest["reference_run"])
    actual = gram_gains(torch.tensor(grams)).numpy()
    train_mean = u[train].mean(0)
    with threadpool_limits(limits=CONFIG["threads"]):
        for arm in NEW_ARMS:
            folder = root / "arms" / arm
            folder.mkdir(parents=True, exist_ok=True)
            fit_path = folder / "fit.npz"
            if not fit_path.exists():
                _event(root, "FITTING", arm=arm)
                model = fit_global(u[train]) if arm == NEW_ARMS[0] else fit_ridge(x[train],u[train],seed=CONFIG["seed"])
                model.save(fit_path)
            else:
                model = GramSimpleGaussian.load(fit_path)
            write_json(folder / "fit_summary.json",model.metadata)
            _event(root, "FIT_COMPLETE", arm=arm)
            coordinate = {p:coordinate_scores(u[ix],model.predict_mean(x[ix]),train_mean) for p,ix in split.items()}
            if "oof_predictions" in model.audit_arrays:
                coordinate["train_nested_penalty_oof"] = coordinate_scores(
                    u[train],model.audit_arrays["oof_predictions"],train_mean)
            write_json(folder / "coordinate_metrics.json", coordinate)
            for number,part in enumerate(PARTITIONS):
                dest = folder / part
                if (dest / "metrics.json").exists():
                    continue
                ix = np.asarray(split[part])
                _event(root, "EVALUATING", arm=arm, partition=part, n=len(ix))
                us = model.sample_coordinates(x[ix],CONFIG["samples"],CONFIG["seed"]+1000+number)
                raw = us*np.asarray(stats["u_scale"])+np.asarray(stats["u_center"])
                sampled = coordinates_to_gram(torch.tensor(raw,dtype=torch.float64)).numpy()
                metadata = dict(arm=arm, target="same standardized nine geometry coordinates as G",
                    training_n=len(train), input="none" if arm==NEW_ARMS[0] else "complete X plus log raw norm",
                    formal_certificate=False, selection="TRAIN-only penalty selection",
                    covariance="full joint; population" if arm==NEW_ARMS[0] else "nested-penalty OOF predictive-error second moment",
                    shared_preprocessing="fixed outer-TRAIN G affine transforms",
                    global_policy_interpretation="uniform subset expectation; lexical ties are arbitrary" if arm==NEW_ARMS[0] else None)
                evaluate_and_save(dest,sampled,grams[ix],ds.ids[ix],metadata=metadata,
                    train_actual_gains=actual[train],score_scale=np.asarray(stats["score_scale"]),
                    seed=CONFIG["seed"], n_bootstrap=CONFIG["n_bootstrap"],n_random=CONFIG["n_random"])
                np.savez_compressed(dest / "coordinate_predictions.npz",ids=ds.ids[ix],
                                    mean=model.predict_mean(x[ix]),actual=u[ix])
                del us, raw, sampled
                _event(root,"PARTITION_COMPLETE",arm=arm,partition=part)
        summarize(root)
    _event(root,"COMPLETE",elapsed_seconds=time.monotonic()-started,final_opened=False,
           fifth_repeat_opened=False,formal_certificate=False)


def summarize(root):
    root = Path(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    reference = Path(manifest["reference_run"])
    paths = {a:root/"arms"/a for a in NEW_ARMS}
    paths.update({a:reference/"arms"/a for a in REFERENCES})
    summaries, paired = {}, {}
    lines = ["# Same-target geometry baseline results", "", "Original opened DEV; original utility and contract unchanged.", "",
        "| Partition | Model | Predicted / actual Gamma | Predicted / observed NULL | CRPS | Brier | Spearman |", "|---|---|---:|---:|---:|---:|---:|"]
    policies = []
    for part in PARTITIONS:
        reports = {a:json.loads((p/part/"metrics.json").read_text()) for a,p in paths.items()}
        summaries[part] = {}
        for arm,report in reports.items():
            act = report["action_metrics"][2]
            pn = report["utility"][2]
            with np.load(paths[arm]/part/"predictions.npz",allow_pickle=False) as values:
                null_predicted = float(values["p_null"][:,2].mean())
            summaries[part][arm] = dict(action=act,crps=pn["crps"],null_predicted=null_predicted,
                energy=report["joint_geometry_energy_score"])
            rho = "undefined (constant)" if act["spearman"] is None else f"{act['spearman']:.4f}"
            lines.append(f"| {part} | {arm} | {act['predicted_mean']:.5f} / {act['actual_mean']:.5f} | {null_predicted:.3f} / {act['null_rate']:.3f} | {pn['crps']:.5f} | {act['null_brier']:.4f} | {rho} |")
            row = next(r for r in report["policy"]["common_budget"] if r["action"]=="Z1Z2" and r["fraction"]==.25 and r["ranking"]=="expected_gain")
            if arm=="GLOBAL_GEOMETRY":
                value=row["matched_random"]["exact_expectation"]
                gain,fdp,fpr=value["expected_per_selected_net_gain"],value["expected_fdp"],value["expected_fpr"]
            else:
                gain,fdp,fpr=row["per_selected_net_gain"],row["fdp"],row["fpr"]
            policies.append(f"| {part} | {arm} | {row['used_wells']} | {row['selected_n']} | {gain:.5f} | {fdp:.3f} | {fpr:.3f} |")
        paired[part] = {}
        for left,right in (("RIDGE_GEOMETRY","GLOBAL_GEOMETRY"),("RIDGE_GEOMETRY","G_DIRECT"),
                           ("RIDGE_GEOMETRY","L_GRAM"),("G_DIRECT","GLOBAL_GEOMETRY")):
            label=left+"__minus__"+right
            pl,pr=paths[left]/part/"predictions.npz",paths[right]/part/"predictions.npz"
            scores=paired_score_comparison(pl,pr,seed=CONFIG["seed"],n_bootstrap=CONFIG["n_bootstrap"])
            policy=paired_policy(reports[left],reports[right],pl,pr,
                left_global=left=="GLOBAL_GEOMETRY",right_global=right=="GLOBAL_GEOMETRY",
                seed=CONFIG["seed"],n_bootstrap=CONFIG["n_bootstrap"])
            paired[part][label] = dict(scores=scores,policy=policy)
    lines += ["", "## ADD_TWO at the 25% physical-well cap", "",
        "All fractions round down. GLOBAL reports exact uniform-subset expectation; no lexical-ID selection skill is claimed.", "",
        "| Partition | Model | Used wells | Selected compounds | Value per selected compound | FDP | FPR |",
        "|---|---|---:|---:|---:|---:|---:|", *policies,
        "", "Full budgets, other actions/rankings, geometry coverage/width and paired score/value/risk intervals are in summary.json and per-arm metrics.",
        "", "The original frozen G preprocessing is shared inside TRAIN CV. OOF dispersion includes fitting uncertainty and bias; it is not an identified pure-noise covariance.",
        "", "Comparisons reuse DEV and shared batches; no new independent holdout, certification or information-theoretic upper bound is claimed.",
        "", "Neighbor support is reported separately in neighbor_support/. No kernel or JEPA is fitted."]
    write_json(root/"summary.json",dict(complete=True,results=summaries,paired=paired,
        formal_certificate=False,final_opened=False,fifth_repeat_opened=False,neural_training=False))
    (root/"REPORT.md").write_text("\n".join(lines)+"\n")


def main(argv=None):
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare"); prep.add_argument("--reference",required=True); prep.add_argument("--output",required=True)
    for name in ("execute","summarize"):
        p=sub.add_parser(name); p.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    if args.command=="prepare":
        prepare(args.reference,args.output)
    elif args.command=="execute":
        execute(args.output)
    else:
        summarize(args.output)


if __name__=="__main__":
    main()
