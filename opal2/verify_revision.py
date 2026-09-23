"""Real already-open DEV engineering verification, not an efficacy experiment."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import numpy as np
import torch

from .config import TrainConfig
from .data import (load_dataset, save_dataset, TrainScaler, fit_library_context, attach_library_context,
                   make_inference_batch)
from .model import MeasurementWorldModel, GroupedProfileEncoder
from .jepa import ConditionalJEPA
from .splits import load_split, save_split
from .training import seed_everything, model_kwargs, fixed_batch, fit_model, load_model
from .evaluation import write_json


def verify(data_directory, output, batch_size=4):
    data_directory, output = Path(data_directory), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config=TrainConfig(batch_size=batch_size,threads=4)
    seed_everything(config.seed,config.threads)
    ds=load_dataset(data_directory/"measurements.npz")
    splits,_=load_split(data_directory/"splits.json",ds.ids)
    scaler=TrainScaler.fit(ds,splits["train"])
    normalized=scaler.transform(ds)
    bank=fit_library_context(normalized,splits["train"])
    normalized=attach_library_context(normalized,bank)
    ix=splits["train"][:batch_size]
    batch,y,mask=fixed_batch(normalized,ix,config)
    model=MeasurementWorldModel(**model_kwargs(ds,config))
    model.set_outcome_transform(scaler.y_center,scaler.y_scale)
    started=time.monotonic()
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    result=model.loss(batch,y,mask)
    result["loss"].backward()
    gradients={}
    for name in ("profile_encoder","chemical_prior","evidence_head","panel_encoder",
                 "library_attention","library_global","operator","mean_head","scale_head","factor_heads"):
        params=list(getattr(model,name).parameters())
        grads=[p.grad for p in params if p.grad is not None]
        gradients[name]={"parameters":sum(p.numel() for p in params),
                         "gradient_tensors":len(grads),
                         "finite":all(torch.isfinite(g).all().item() for g in grads),
                         "absolute_gradient_sum":sum(g.abs().sum().item() for g in grads)}
        if not grads or not gradients[name]["finite"] or gradients[name]["absolute_gradient_sum"]<=0:
            raise AssertionError(f"Real-data training path inactive/nonfinite: {name}")
    torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip,error_if_nonfinite=True)
    optimizer.step()
    seconds=time.monotonic()-started
    report={"purpose":"REAL_DEV_ENGINEERING_TEST_NOT_TRAINED_PERFORMANCE",
            "data":str(data_directory.resolve()),"shape":list(ds.Y.shape),
            "gradient_batch_size":batch_size,"retained_outcome_coordinates":ds.Y.shape[-1],
            "real_reference_controls":len(ds.panel_y),"configuration":model.config,
            "full_parameter_count":sum(p.numel() for p in model.parameters()),
            "full_forward_backward_optimizer_seconds":seconds,"gradient_paths":gradients,
            "initial_loss_components":{k:float(v.detach()) for k,v in result.items() if torch.is_tensor(v) and v.numel()==1}}
    if report["initial_loss_components"]["reference_reconstruction_count"]<=0:
        raise AssertionError("Real reference reconstruction was not exercised")
    model.eval()
    with torch.no_grad():
        before=model(batch)
        # Poison only future treatment measurements. Reconstruct every input.
        old=normalized.Y[ix,1:4].copy()
        normalized.Y[ix,1:4]=1e9
        changed,_,_=fixed_batch(normalized,ix,config)
        after=model(changed)
        normalized.Y[ix,1:4]=old
        for name in ("mean","diag_var","factors"):
            torch.testing.assert_close(getattr(before,name),getattr(after,name),rtol=0,atol=0)
        zero=make_inference_batch(normalized,ix,(),(0,1,2,3),reference_access=config.reference_access)
        distribution=model(zero)
        assert torch.isfinite(distribution.mean).all()
        components=model.variance_components(batch)
        report["variance_components"]={k:({"mean":float(v.mean()),
                  "per_well_feature_mean":v.mean(-1).cpu().numpy() if v.ndim>=3 else v.cpu().numpy()}
                  if torch.is_tensor(v) else v) for k,v in components.items()}
        report["variance_component_units"]="model-standardized coordinates; not causal variance fractions"
    state_path=output/"engineering_step.pt"
    torch.save({"state_dict":model.state_dict(),"model_config":model.config,"training_steps":1},state_path)
    restored=MeasurementWorldModel(**model.config).eval()
    restored.load_state_dict(torch.load(state_path,weights_only=True)["state_dict"],strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(batch).mean,before.mean,rtol=0,atol=0)
    del restored, model, optimizer, before, after, distribution, result
    gc.collect()
    encoder=GroupedProfileEncoder(ds.feature_groups,config.hidden_dim,
                                  attention_layers=config.group_attention_layers,attention_heads=config.attention_heads)
    learner=ConditionalJEPA(encoder,condition_dim=ds.dimensions["K"],reference_dim=ds.dimensions["R"],
                            chemical_dim=ds.dimensions["H"],hidden_dim=config.hidden_dim)
    started=time.monotonic()
    result=learner.loss(batch,y,mask)
    result["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in learner.parameters() if p.grad is not None)
    learner.update_teacher()
    report["full_jepa_forward_backward_ema_seconds"]=time.monotonic()-started
    report.update(future_outcome_poison_test=True,zero_context_test=True,checkpoint_reload_exact=True,
                  original_final_opened=False,fifth_repeat_opened=False,reference_target2_available=False,
                  status="PASS_FULL_ARCHITECTURE_ENGINEERING_CHECK")
    write_json(output/"repair_verification.json",report)
    return report


def training_roundtrip(data_directory, output):
    """Full-length training on a declared small, already-open real DEV cohort.

    All model modules, 3617 coordinates, 60 JEPA epochs and the 100-epoch
    validation-stopped probabilistic schedule are retained. The small cohort
    tests software execution only; it does not support an efficacy conclusion.
    """
    from .evaluation import evaluate_model
    data_directory,output=Path(data_directory),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    ds=load_dataset(data_directory/"measurements.npz")
    source_splits,_=load_split(data_directory/"splits.json",ds.ids)
    counts={"train":38,"validation":10,"calibration":6,"evaluation":10}
    picked={name:source_splits[name][:count] for name,count in counts.items()}
    indices=np.concatenate(list(picked.values()))
    small=ds.subset(indices)
    lookup={int(old):new for new,old in enumerate(indices)}
    splits={name:np.array([lookup[int(i)] for i in values]) for name,values in picked.items()}
    config=TrainConfig()
    manifest={"purpose":"SMALL_REAL_DATA_COMPLETE_TRAINING_ROUNDTRIP_NOT_EFFICACY",
              "selection":"first IDs of each already-frozen old DEV partition, no outcomes used",
              "compound_ids":{name:small.ids[ix].tolist() for name,ix in splits.items()},
              "shape":list(small.Y.shape),"epochs_max":config.epochs,"jepa_epochs":config.jepa_epochs,
              "full_model_width":config.hidden_dim,"latent_rank":config.latent_rank,
              "residual_rank":config.residual_rank,"old_final_opened":False,"fifth_repeat_opened":False}
    write_json(output/"run_manifest.json",manifest)
    save_dataset(small,output/"data"/"measurements.npz")
    save_split(output/"data"/"splits.json",small.ids,splits,evidence_scope=manifest["purpose"])
    fit_model(small,splits,config,output/"D")
    model,scaler,reloaded,checkpoint=load_model(output/"D")
    result=evaluate_model(model,scaler,small,splits,reloaded,output/"D")
    result["purpose"]=manifest["purpose"]
    result["checkpoint_epoch"]=checkpoint["epoch"]
    write_json(output/"repair_verification.json",result)
    print({"status":"COMPLETE_FULL_TRAINING_ROUNDTRIP","epoch":checkpoint["epoch"],"output":str(output)},flush=True)
    return result


def workflow_check(data_directory, engineering_directory, output):
    """Exercise the connected decision pipeline on real DEV, without certification.

    Uses the full one-optimizer-step engineering checkpoint, not a trained
    scientific model. The simulation count is an explicit software-test setting;
    its pass proportion is not an assurance estimate for a real study.
    """
    from .decision_workflow import run_model_decision_workflow
    from .provenance import bind_fitting_provenance
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    ds=load_dataset(Path(data_directory)/"measurements.npz")
    splits,_=load_split(Path(data_directory)/"splits.json",ds.ids)
    config=TrainConfig()
    seed_everything(config.seed,config.threads)
    scaler=TrainScaler.fit(ds,splits["train"])
    payload=torch.load(Path(engineering_directory)/"engineering_step.pt",weights_only=True)
    model=MeasurementWorldModel(**payload["model_config"]).eval()
    model.load_state_dict(payload["state_dict"],strict=True)
    bind_fitting_provenance(model,ds.ids[splits["train"]],ds.ids[splits["validation"]])
    model.library_bank=fit_library_context(scaler.transform(ds),splits["train"])
    settings={"purpose":"REAL_DEV_ONE_STEP_CHECKPOINT_WORKFLOW_ENGINEERING_ONLY",
        "diagnostic_indices":splits["evaluation"][:2].tolist(),"samples":config.samples,
        "assume_iid_campaigns":False,"planner":{"budget":1,"risk_penalty":.02},
        "contract":{"min_mean_net_gain":0.,"max_mean_wells":2.,"alpha":.05},
        "assurance":{"simulations":4,"campaign_counts":[2,3],"minimum_campaigns":2}}
    write_json(output/"run_manifest.json",settings)
    result=run_model_decision_workflow(model,scaler,ds,config,settings,output)
    assert result["status"]=="MODEL_NOT_ADMITTED"
    assert result["development_diagnostic"]["is_authorized"] is False
    assert result["model_conditional_assurance"]["empirical_admission_overridden"] is False
    write_json(output/"repair_verification.json",{
        "status":"PASS_REAL_MODEL_WORKFLOW_CONNECTION_CHECK",
        "purpose":settings["purpose"],"future_spectrum_MC_samples":config.samples,
        "joint_model_assurance_executed":True,"simulation_replicates_for_software_check_only":4,
        "empirical_model_admission":"BLOCKED_EXPECTED_SINGLE_SOURCE",
        "scientific_effectiveness_test":False,"old_final_opened":False,"fifth_repeat_opened":False})
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--training-roundtrip",action="store_true")
    parser.add_argument("--workflow-checkpoint",help="Full one-step engineering checkpoint directory for the connected workflow check")
    args=parser.parse_args()
    if args.workflow_checkpoint:
        result=workflow_check(args.data,args.workflow_checkpoint,args.output)
        print({"status":"PASS_REAL_MODEL_WORKFLOW_CONNECTION_CHECK","admission":result["status"]},flush=True)
    elif args.training_roundtrip:
        training_roundtrip(args.data,args.output)
    else:
        result=verify(args.data,args.output,args.batch_size)
        print({"status":result["status"],"full_parameter_count":result["full_parameter_count"],
               "optimizer_step_seconds":result["full_forward_backward_optimizer_seconds"]},flush=True)
