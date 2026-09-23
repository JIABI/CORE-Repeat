"""Postprocessing plus a separately labelled, unchanged-law numerical diagnostic.

The strict original decoder's calibration stop is retained. No model is refit,
no coordinate/sample is clipped, and no held-out error changes a fitted value.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .biology_kernel_experiment import _load_study_data
from .biology_kernel_evaluation import write_json
from .data import TrainScaler
from .gram_evaluation import evaluate_and_save, paired_score_comparison
from .gram_geometry import profiles_to_gram, gram_gains
from .gram_simple_models import GramSimpleGaussian
from .gram_simple_experiment import paired_policy, PARTITIONS, CONFIG, NEW_ARMS, REFERENCES


def numerical_calibration_followup(root):
    from .gram_forward_diagnostic import coordinates_factor_forward, factor_forward_consistency
    root = Path(root).resolve()
    record = root/"numerical_followup"
    record.mkdir(exist_ok=True)
    output = root/"arms/RIDGE_GEOMETRY/calibration_forward_diagnostic"
    if (output/"metrics.json").exists():
        return
    manifest = json.loads((root/"run_manifest.json").read_text())
    if manifest["config"] != CONFIG:
        raise ValueError("Original sampling configuration changed")
    for key in ("final_opened", "fifth_repeat_opened", "original_endpoint_changed",
                "original_contract_changed", "original_split_changed"):
        if manifest[key] is not False:
            raise ValueError("Original scope changed")
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        diagnostic="factor-forward evaluation after strict inverse-Schur numerical stop",
        primary_stop_retained=True, partition="calibration", refit=False,
        original_samples=2000, original_seed=CONFIG["seed"]+1002,
        changed_fitted_parameters=False, clipping=False, jitter=False,
        removed_samples=0, removed_compounds=0, changed_endpoint=False,
        interpretation="same mathematical forward map; not a strict invertibility pass or a certification",
        source_files=["opal2/gram_simple_followup.py","opal2/gram_forward_diagnostic.py"])
    write_json(record/"PROTOCOL.json",protocol)
    for name in protocol["source_files"]:
        source = Path(__file__).resolve().parents[1]/name
        shutil.copy2(source,record/source.name)
    ds, split, _ = _load_study_data(manifest["data_directory"])
    if {k:ds.ids[v].tolist() for k,v in split.items()} != manifest["compound_ids"]:
        raise ValueError("Original compound identities changed")
    stats = json.loads((root/"preprocessing.json").read_text())
    scaler = TrainScaler.load(root/"input_scaler.json")
    if scaler.feature_names!=ds.feature_names.tolist() or scaler.train_ids!=manifest["compound_ids"]["train"]:
        raise ValueError("Original scaler changed")
    ix=np.asarray(split["calibration"])
    xx=scaler.transform_y(ds.Y[ix,0]).astype(np.float32).astype(np.float64)
    lognorm=np.log(np.linalg.norm(ds.Y[ix,0],axis=-1))
    nn=((lognorm-stats["lognorm_center"])/stats["lognorm_scale"]).astype(np.float32).astype(np.float64)
    x=np.column_stack((xx,nn))
    model=GramSimpleGaussian.load(root/"arms/RIDGE_GEOMETRY/fit.npz")
    us=model.sample_coordinates(x,CONFIG["samples"],CONFIG["seed"]+1002)
    raw=us*np.asarray(stats["u_scale"])+np.asarray(stats["u_center"])
    u=torch.tensor(raw,dtype=torch.float64)
    grams, diagnostic=coordinates_factor_forward(u)
    consistency=factor_forward_consistency(u,grams)
    if consistency["gains_max_absolute_error"]>1e-10 or consistency["observables_max_absolute_error"]>1e-8:
        raise ValueError("Factor/Gram forward identities do not agree at numerical precision")
    counts=np.asarray(diagnostic["recovered_schur_failed_mask"],dtype=bool).sum(0)
    record_diagnostic={k:v for k,v in diagnostic.items() if k not in (
        "recovered_schur_failed_mask", "recovered_schur_failed_indices", "recovered_schur_info")}
    record_diagnostic.update(failed_compounds=[dict(compound_id=str(ds.ids[ix[i]]), failed_draws=int(counts[i]))
                                              for i in np.flatnonzero(counts)],
        sample_count=CONFIG["samples"], compounds=len(ix), total_draw_objects=len(ix)*CONFIG["samples"],
        consistency={k:v for k,v in consistency.items() if not k.endswith("per_draw_object")},
        no_parameter_or_sample_changes=True,
        primary_strict_decoder_status="NUMERICAL_STOP", future_calibration_targets_used_to_fit=False)
    np.savez_compressed(record/"inverse_schur_failure_mask.npz", ids=ds.ids[ix],
                        failed=np.asarray(diagnostic["recovered_schur_failed_mask"],dtype=bool),
                        info=diagnostic["recovered_schur_info"],
                        gain_absolute_error=consistency["gain_absolute_error_per_draw_object"],
                        observable_absolute_error=consistency["observable_absolute_error_per_draw_object"])
    write_json(record/"diagnostic.json",record_diagnostic)
    # Exact original cached target arrays prevent a change of evaluation geometry.
    reference=Path(manifest["reference_run"])
    with np.load(reference/"arms/L_GRAM/calibration/predictions.npz",allow_pickle=False) as saved:
        if not np.array_equal(ds.ids[ix],saved["ids"]) or not np.array_equal(stats["score_scale"],saved["score_scale"]):
            raise ValueError("Original target identity/metric mismatch")
        target=saved["actual_grams"].copy()
    training_grams=profiles_to_gram(torch.tensor(ds.Y[split["train"]],dtype=torch.float64))
    training_actual=gram_gains(training_grams).numpy()
    evaluate_and_save(output,grams.numpy(),target,ds.ids[ix],
        metadata=dict(arm="RIDGE_GEOMETRY",numerical_diagnostic=True,
            strict_decoder_pass=False, same_parameters=True,same_samples=True,
            refit=False,endpoint_changed=False,formal_certificate=False),
        train_actual_gains=training_actual,score_scale=np.asarray(stats["score_scale"]),
        seed=CONFIG["seed"],n_bootstrap=CONFIG["n_bootstrap"],n_random=CONFIG["n_random"])


def summarize(root):
    root=Path(root).resolve()
    manifest=json.loads((root/"run_manifest.json").read_text())
    reference=Path(manifest["reference_run"])
    results,paired={},{}
    headline=["# Same-target geometry baselines and local-support results", "",
        "The two models were fitted on TRAIN only. The strict RIDGE calibration decoder stopped; a separately labelled same-parameter, same-sample factor-forward diagnostic is included below. No outcome or sample was removed.", "",
        "| Partition | Model | Predicted / actual Gamma | Predicted / observed NULL | CRPS | Brier | Spearman |",
        "|---|---|---:|---:|---:|---:|---:|"]
    policy_lines=[]
    for part in PARTITIONS:
        paths={a:(root if a in NEW_ARMS else reference)/"arms"/a/part for a in (*NEW_ARMS,*REFERENCES)}
        supplement=part=="calibration"
        if supplement:
            paths["RIDGE_GEOMETRY"]=root/"arms/RIDGE_GEOMETRY/calibration_forward_diagnostic"
        reports={a:json.loads((p/"metrics.json").read_text()) for a,p in paths.items()}
        results[part]={}
        for arm,m in reports.items():
            a=m["action_metrics"][2]
            with np.load(paths[arm]/"predictions.npz",allow_pickle=False) as vals:
                pn=float(vals["p_null"][:,2].mean())
            diagnostic=supplement and arm=="RIDGE_GEOMETRY"
            results[part][arm]=dict(action=a,crps=m["utility"][2]["crps"],null_predicted=pn,
                energy=m["joint_geometry_energy_score"],numerical_forward_diagnostic=diagnostic)
            rho="undefined (constant)" if a["spearman"] is None else f"{a['spearman']:.4f}"
            label=arm+(" [forward diagnostic]" if diagnostic else "")
            headline.append(f"| {part} | {label} | {a['predicted_mean']:.5f} / {a['actual_mean']:.5f} | {pn:.3f} / {a['null_rate']:.3f} | {m['utility'][2]['crps']:.5f} | {a['null_brier']:.4f} | {rho} |")
            row=next(r for r in m["policy"]["common_budget"] if r["action"]=="Z1Z2" and r["fraction"]==.25 and r["ranking"]=="expected_gain")
            if arm=="GLOBAL_GEOMETRY":
                v=row["matched_random"]["exact_expectation"]
                gain,fdp,fpr=v["expected_per_selected_net_gain"],v["expected_fdp"],v["expected_fpr"]
            else:
                gain,fdp,fpr=row["per_selected_net_gain"],row["fdp"],row["fpr"]
            policy_lines.append(f"| {part} | {label} | {row['used_wells']} | {row['selected_n']} | {gain:.5f} | {fdp:.3f} | {fpr:.3f} |")
        paired[part]={}
        for left,right in (("RIDGE_GEOMETRY","GLOBAL_GEOMETRY"),("RIDGE_GEOMETRY","G_DIRECT"),
                           ("RIDGE_GEOMETRY","L_GRAM"),("G_DIRECT","GLOBAL_GEOMETRY")):
            pl,pr=paths[left]/"predictions.npz",paths[right]/"predictions.npz"
            paired[part][left+"__minus__"+right]=dict(
                scores=paired_score_comparison(pl,pr,seed=CONFIG["seed"],n_bootstrap=CONFIG["n_bootstrap"]),
                policy=paired_policy(reports[left],reports[right],pl,pr,right_global=right=="GLOBAL_GEOMETRY",
                    seed=CONFIG["seed"],n_bootstrap=CONFIG["n_bootstrap"]),
                includes_numerical_forward_diagnostic=supplement and (left=="RIDGE_GEOMETRY" or right=="RIDGE_GEOMETRY"))
    headline += ["", "## ADD_TWO at the 25% physical-well cap", "",
        "GLOBAL values are uniform-subset expectations. Lexical-ID tie-break outcomes in raw evaluator files are not selection skill.", "",
        "| Partition | Model | Used wells | Selected compounds | Value per selected compound | FDP | FPR |",
        "|---|---|---:|---:|---:|---:|---:|",*policy_lines,
        "", "All 5/10/25% budgets, two ranking rules and three original actions remain in the per-arm metrics. Paired scores and value/risk intervals are in summary.json.",
        "", "## Numerical diagnostic", "",
        "The original strict decoder and its failed calibration result were not changed. See numerical_followup/diagnostic.json for the factor-forward identity check, failed draw/object counts and no-refit record.",
        "The factor-forward diagnostic uses the identical mathematical distribution and original samples; it is not a strict numerical invertibility pass, an added covariance floor, or a statistical certificate.",
        "", "## Scope", "",
        "TRAIN-internal nested penalty selection shares the previously fixed G preprocessing. OOF covariance contains predictive error and bias, not identified pure measurement noise. No new model was selected on held-out outcomes.",
        "These are reused DEV partitions with shared batches. Pairwise intervals condition on fitted predictions and do not include model search or certify new environments.",
        "No FINAL, fifth repeat, additional compound, kernel training, JEPA training, original endpoint or seven-item contract was changed.",
        "", "Neighbor-support results: [report](neighbor_support/REPORT.md). Similarity availability is not evidence that neighbors share response or noise."]
    numerical=json.loads((root/"numerical_followup/diagnostic.json").read_text())
    write_json(root/"summary.json",dict(analysis_complete=True,primary_all_partitions_completed=False,
        numerical_followup_complete=True,results=results,paired=paired,numerical_diagnostic=numerical,
        final_opened=False,fifth_repeat_opened=False,formal_certificate=False))
    (root/"REPORT.md").write_text("\n".join(headline)+"\n")
    write_json(root/"status.json",dict(utc=datetime.now(timezone.utc).isoformat(),
        state="ANALYSIS_COMPLETE_WITH_NUMERICAL_DIAGNOSTIC",primary_strict_calibration="NUMERICAL_STOP",
        numerical_followup="COMPLETE",final_opened=False,fifth_repeat_opened=False,formal_certificate=False))


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    torch.set_num_threads(CONFIG["threads"])
    with threadpool_limits(limits=CONFIG["threads"]):
        numerical_calibration_followup(args.output)
        summarize(args.output)
    print(json.dumps(dict(state="ANALYSIS_COMPLETE_WITH_NUMERICAL_DIAGNOSTIC",output=args.output)),flush=True)


if __name__=="__main__":
    main()
