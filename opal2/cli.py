"""Reproducible command-line entry points for full real-measurement experiments."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from .config import TrainConfig
from .data import load_dataset, save_dataset
from .splits import development_split, load_split, save_split


ARMS = {
    "A": ("mlp", False),
    "B": ("measurement", False),
    "C": ("mlp", True),
    "D": ("measurement", True),
    "G": ("generic", False),
    "H": ("generic", True),
    "FR": ("measurement", False),
    "JF": ("measurement", True),
    "NO_CHEM": ("measurement", True),
    "NO_REF": ("measurement", True),
    "NO_LIBRARY": ("measurement", True),
    "NO_GROUP_INTERACTION": ("measurement", True),
    "SHUFFLED_GROUPS": ("measurement", True),
}


def arm_config(base, arm):
    mode, jepa = ARMS[arm]
    changes = dict(kernel_mode=mode, use_jepa=jepa, encoder_policy="auto")
    if arm == "FR":
        changes["encoder_policy"] = "frozen_random"
    elif arm == "JF":
        changes["encoder_policy"] = "jepa_finetune"
    elif arm == "NO_CHEM":
        changes["use_chemistry"] = False
    elif arm == "NO_REF":
        changes["use_references"] = False
    elif arm == "NO_LIBRARY":
        changes["use_library"] = False
    elif arm == "NO_GROUP_INTERACTION":
        changes["group_attention_layers"] = 0
    elif arm == "SHUFFLED_GROUPS":
        changes["group_assignment"] = "shuffled"
    return replace(base, **changes).validate()


def prepare(args):
    from .source5 import load_source5
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    dataset = load_source5(args.space, legacy_root=args.legacy_root)
    save_dataset(dataset, directory / "measurements.npz")
    splits = development_split(dataset.ids, args.seed)
    save_split(directory / "splits.json", dataset.ids, splits,
               evidence_scope="DEVELOPMENT_ONLY_ALREADY_OPEN_639_SOURCE5")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "shape": list(dataset.Y.shape), "space": args.space,
                "split_counts": {k: len(v) for k, v in splits.items()},
                "fifth_repeat_read": False, "old_final_opened": False,
                "endpoint": "original fixed-space half-cosine gain, unchanged",
                "actions": ["STOP", "ADD_Z1", "ADD_Z2", "ADD_Z1_Z2"],
                "cost_per_optional_well": .01, "positive_margin": .005,
                "reference_access": "observed_only",
                "selection": "validation likelihood only; evaluation is not for choosing an arm",
                "model_arms": {k: {"kernel_mode": v[0], "use_jepa": v[1]} for k, v in ARMS.items()},
                "limits": "Previously explored compounds, shared known batches; development evaluation only"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def get_data(args):
    dataset = load_dataset(Path(args.data) / "measurements.npz")
    splits, scope = load_split(Path(args.data) / "splits.json", dataset.ids)
    return dataset, splits


def attach_biology(args):
    """Attach curated sidecar metadata to a NEW portable export, without fitting."""
    from .biology import load_biology, validate_records
    dataset = load_dataset(args.dataset)
    records = validate_records(load_biology(args.annotations), dataset.ids, dataset.well_ids)
    updated = replace(dataset, biology_records=records, biology_vocabulary=None)
    path, sidecar = save_dataset(updated, args.output)
    print(json.dumps({"dataset": str(path), "sidecar": str(sidecar),
                      "units": len(records), "input_relations": sum(r.usable for record in records for r in record.relations),
                      "fitted_vocabulary": False, "training_started": False}), flush=True)


def experiment(args):
    from .training import fit_model, load_model
    from .evaluation import evaluate_model, write_json
    dataset, splits = get_data(args)
    base = TrainConfig.load(args.config) if args.config else TrainConfig()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in args.arms:
        config = arm_config(base, arm)
        destination = output / arm
        model, scaler = fit_model(dataset, splits, config, destination, resume=args.resume)
        # Evaluation deliberately uses the saved/reloaded checkpoint, not a
        # second in-memory object with possibly different training state.
        model, scaler, config, payload = load_model(destination)
        results[arm] = evaluate_model(model, scaler, dataset, splits, config, destination)
        write_json(output / ("completed_arms_" + "_".join(args.arms) + ".json"), results)
        print(json.dumps({"completed_arm": arm, "best_epoch": payload["epoch"],
                          "evaluation": results[arm]}, indent=2), flush=True)


def train(args):
    from .training import fit_model
    dataset, splits = get_data(args)
    config = TrainConfig.load(args.config) if args.config else TrainConfig()
    fit_model(dataset, splits, config, args.output, resume=args.resume)


def evaluate(args):
    from .training import load_model, seed_everything
    from .evaluation import evaluate_model
    dataset, splits = get_data(args)
    model, scaler, config, _ = load_model(args.model)
    seed_everything(config.seed, config.threads)
    summary = evaluate_model(model, scaler, dataset, splits, config, args.model)
    print(json.dumps(summary, indent=2), flush=True)


def original_baseline(args):
    from .legacy_baseline import run_legacy_baseline
    from .evaluation import clean
    dataset, splits = get_data(args)
    result = run_legacy_baseline(dataset, splits, args.output, args.legacy_root)
    print(json.dumps(clean({"model": result["model"], "gain_metrics": result["gain_metrics"]}), indent=2), flush=True)


def probe(args):
    from .probe import run_model_probe, evaluate_observed_probe
    from .training import load_model, seed_everything
    from .evaluation import write_json
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Use a new result filename")
    dataset, splits = get_data(args)
    model, scaler, config, _ = load_model(args.model)
    seed_everything(config.seed, config.threads)
    indices = splits["evaluation"]
    roles = (0, args.probe_role, 3 - args.probe_role, 3)
    common = dict(total_budget=args.budget, selection_samples=args.selection_samples,
                  evaluation_samples=args.evaluation_samples, reference_access=config.reference_access,
                  seed=args.seed, roles=roles, update_mode=args.update_mode)
    if args.mode == "predicted":
        result = run_model_probe(model, scaler, dataset, indices,
                                 outer_samples=args.outer_samples, **common)
    else:
        result = evaluate_observed_probe(model, scaler, dataset, indices, **common)
    write_json(destination, result)
    print(json.dumps({"mode": args.mode, "output": str(destination),
                      "evaluation_compounds": len(indices), "budget": args.budget}), flush=True)


def decision_workflow(args):
    from .decision_workflow import run_model_decision_workflow
    from .training import load_model, seed_everything
    dataset, _ = get_data(args)
    model, scaler, config, _ = load_model(args.model)
    seed_everything(config.seed, config.threads)
    settings = json.loads(Path(args.settings).read_text())
    result = run_model_decision_workflow(model, scaler, dataset, config, settings, args.output)
    print(json.dumps(result, indent=2, default=str), flush=True)


def selective_probe(args):
    from .probe import run_selective_model_probe
    from .training import load_model, seed_everything
    from .evaluation import write_json
    dataset,splits=get_data(args)
    model,scaler,config,_=load_model(args.model)
    seed_everything(config.seed,config.threads)
    output=Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new selective-probe result path")
    planning = json.loads(Path(args.planning_settings).read_text()) if args.planning_settings else {}
    allowed = {"target_costs", "setup_memberships", "setup_costs", "cost_budget", "cost_per_well",
               "max_total_null_fraction", "max_incremental_null_fraction", "risk_penalty", "missing_outcome"}
    if not isinstance(planning,dict) or set(planning)-allowed:
        raise ValueError("Unknown selective-probe planning setting; use declared cost/risk/missing fields only")
    if args.risk_penalty is not None:
        planning["risk_penalty"] = args.risk_penalty
    result=run_selective_model_probe(model,scaler,dataset,splits["evaluation"],
        total_budget=args.budget,lookahead_depth=args.lookahead_depth,
        outer_samples=args.outer_samples,selection_samples=config.samples,evaluation_samples=config.samples,
        roles=(0,args.probe_role,3-args.probe_role,3),reference_access=config.reference_access,
        seed=config.seed,retrospective=args.observed,**planning)
    write_json(output,result)
    print(json.dumps({"output":str(output),"mode":result["mode"]}),flush=True)


def cross_validate(args):
    from .experiments import nested_development_cv
    dataset, _ = get_data(args)
    base = TrainConfig.load(args.config) if args.config else TrainConfig()
    settings = {name: arm_config(base,name) for name in args.arms}
    groups = None
    if args.group_labels:
        mapping = json.loads(Path(args.group_labels).read_text())
        groups = [mapping[str(compound)] for compound in dataset.ids]
    nested_development_cv(dataset, settings, args.output, outer_folds=args.outer_folds,
                          inner_folds=args.inner_folds, groups=groups, seed=base.seed)


def prepare_source_holdout(args):
    import numpy as np
    from .splits import materialize_source_partitions, merge_compound_disjoint_partitions
    ds=load_dataset(args.dataset)
    spec=json.loads(Path(args.partition).read_text())
    # These are caller-supplied actual well-source labels, not guessed from a
    # local one-hot column or invented domains in a single-source DEV archive.
    labels=np.asarray(spec["well_sources"],str)
    parts=materialize_source_partitions(ds,labels,
        train_sources=spec["train_sources"],validation_sources=spec["validation_sources"],
        calibration_sources=spec["calibration_sources"],evaluation_sources=spec["evaluation_sources"],
        minimum_wells=4,allow_known_compounds=False)
    merged,splits=merge_compound_disjoint_partitions(parts)
    destination=Path(args.output)
    destination.mkdir(parents=True,exist_ok=False)
    save_dataset(merged,destination/"measurements.npz")
    save_split(destination/"splits.json",merged.ids,splits,
               evidence_scope="DECLARED_SOURCE_AND_COMPOUND_HOLDOUT_FOUR_ROLE_TASK")
    (destination/"manifest.json").write_text(json.dumps({"partition":spec,
        "scope":"new source and new compounds; known-compound target-domain experiment not conflated",
        "split_counts":{k:len(v) for k,v in splits.items()},"minimum_wells_per_compound":4},indent=2)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="opal2")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("attach-biology", help="Attach typed biological annotations to a NEW dataset export; never train")
    p.add_argument("--dataset", required=True, help="Existing portable NPZ with matching JSON sidecar")
    p.add_argument("--annotations", required=True, help="Biology schema-v1 JSON covering exactly these units")
    p.add_argument("--output", required=True, help="New NPZ destination; existing paths are not overwritten")
    p.set_defaults(func=attach_biology)
    p = sub.add_parser("prepare-source5", help="Import only the previously opened 639 DEV, four roles")
    p.add_argument("--legacy-root", required=True)
    p.add_argument("--space", choices=["primary", "spatial"], default="primary")
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--output", required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("original-baseline", help="Refit the original complete OPAL three heads")
    p.add_argument("--legacy-root", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=original_baseline)
    p = sub.add_parser("probe", help="Legacy all-P comparator; use selective-probe for first-stage selection")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["predicted", "observed"], required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--probe-role", type=int, choices=[1, 2], default=1)
    p.add_argument("--update-mode", choices=["gaussian_condition", "reencode"],
                   default="gaussian_condition")
    p.add_argument("--outer-samples", type=int, default=16)
    p.add_argument("--selection-samples", type=int, default=2000)
    p.add_argument("--evaluation-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260911)
    p.set_defaults(func=probe)
    p = sub.add_parser("decision-workflow", help="Model admission, frozen-family selection, sequential contract and assurance")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--settings", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=decision_workflow)
    p=sub.add_parser("selective-probe",help="Choose stop, direct acquisition, or which first information probe to buy")
    p.add_argument("--data",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--budget",type=int,required=True)
    p.add_argument("--probe-role",type=int,choices=[1,2],default=1)
    p.add_argument("--lookahead-depth",type=int,default=1)
    p.add_argument("--outer-samples",type=int,default=16)
    p.add_argument("--risk-penalty",type=float,default=None,help="Optional override of the declared planning-settings penalty")
    p.add_argument("--planning-settings",help="JSON candidate costs, setup memberships/costs, monetary budget and model-risk limits")
    p.add_argument("--observed",action="store_true",help="Reveal only selected probes; score V only after decisions")
    p.set_defaults(func=selective_probe)
    p = sub.add_parser("cross-validate", help="Full nested development comparison; never a FINAL certificate")
    p.add_argument("--data", required=True)
    p.add_argument("--config")
    p.add_argument("--arms", nargs="+", choices=list(ARMS), required=True)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--group-labels", help="JSON mapping compound IDs to independent development groups")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cross_validate)
    p=sub.add_parser("prepare-source-holdout",help="Physical source/compound partition before pairing and fitting")
    p.add_argument("--dataset",required=True,help="Portable actual measurement NPZ")
    p.add_argument("--partition",required=True,help="JSON actual well-source labels and four source lists")
    p.add_argument("--output",required=True)
    p.set_defaults(func=prepare_source_holdout)
    for name in ("experiment", "train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True, help="Prepared dataset directory")
        if name == "evaluate":
            p.add_argument("--model", required=True)
            p.set_defaults(func=evaluate)
        else:
            p.add_argument("--config")
            p.add_argument("--output", required=True)
            p.add_argument("--resume", action="store_true")
            if name == "experiment":
                p.add_argument("--arms", nargs="+", choices=list(ARMS), default=["A", "B", "C", "D"])
                p.set_defaults(func=experiment)
            else:
                p.set_defaults(func=train)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
