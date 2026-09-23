"""Paired JEPA optimization diagnostics, separate from formal model training.

Two learning-rate schedules start at identical fresh weights and see the same
compound episodes and random streams. No measurement-world-model fit or
calibration/evaluation-outcome access is performed by this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from .jepa import ConditionalJEPA
from .splits import assert_disjoint_splits
from .training import fixed_batch, random_training_batch, seed_everything


ARM_NAMES=("R1_COSINE_3E4","R2_COSINE_1E4")


@dataclass
class DiagnosticConfig:
    epochs: int = 60
    batch_size: int = 128
    report_every: int = 5
    eval_chunk_size: int = 16
    profile_chunk_size: int = 16
    base_lrs: tuple[float,float] = (3e-4,1e-4)
    min_lr_ratio: float = .01
    ema_reference_decay: float = .99
    ema_reference_batch: int = 32
    diagnostic_compounds: int = 96
    gradient_clip: float = 5.0
    weight_decay: float = 1e-4

    def validate(self):
        for name in ("epochs","batch_size","report_every","eval_chunk_size","profile_chunk_size",
                     "ema_reference_batch","diagnostic_compounds"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.base_lrs)!=2 or any(not math.isfinite(x) or x<=0 for x in self.base_lrs):
            raise ValueError("Two positive finite diagnostic learning rates are required")
        self.base_lrs=tuple(self.base_lrs)
        if not 0<self.min_lr_ratio<=1 or not 0<self.ema_reference_decay<1:
            raise ValueError("Invalid cosine endpoint or sample-equivalent EMA")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip<=0 or not math.isfinite(self.weight_decay) or self.weight_decay<0:
            raise ValueError("Invalid clipping or weight decay")
        return self


def cosine_learning_rate(base_lr, minimum_lr, step, total_steps):
    """Cosine endpoints include the first and final actual optimizer steps."""
    if (not isinstance(step,int) or not isinstance(total_steps,int) or total_steps<1
            or not 0<=step<total_steps or not 0<minimum_lr<=base_lr):
        raise ValueError("Invalid finite cosine schedule")
    fraction=step/(total_steps-1) if total_steps>1 else 0.0
    return minimum_lr+.5*(base_lr-minimum_lr)*(1+math.cos(math.pi*fraction))


def exposure_matched_ema(batch_n, reference_decay=.99, reference_batch=32):
    if batch_n<1 or reference_batch<1 or not 0<reference_decay<1:
        raise ValueError("Invalid sample-exposure EMA")
    return reference_decay**(batch_n/reference_batch)


@contextmanager
def preserve_randomness():
    """Diagnostics cannot advance any subsequent training random stream."""
    python_state=random.getstate();numpy_state=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)


@contextmanager
def _training_stream(seed,epoch,batch):
    with preserve_randomness():
        torch.manual_seed((seed+7919*(epoch+1)+104729*(batch+1)) % (2**63-1))
        yield


def _atomic_json(path, value):
    path=Path(path); temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _atomic_torch(path, value):
    path=Path(path);temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _record(path, event, **value):
    entry={"utc":datetime.now(timezone.utc).isoformat(),"event":event,**value}
    with Path(path).open("a") as stream:
        stream.write(json.dumps(entry,allow_nan=False)+"\n");stream.flush()
    print(json.dumps(entry,allow_nan=False),flush=True)
    return entry


def _status(directory,state,phase,*,epoch=None,arm=None,**extra):
    _atomic_json(Path(directory)/"status.json",dict(state=state,phase=phase,epoch=epoch,arm=arm,
                 pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**extra))


def _make_learners(train_config,kwargs,config,encoder_class=None):
    if encoder_class is None:
        from .jepa_memory import MemoryEfficientGroupedProfileEncoder
        encoder_class=MemoryEfficientGroupedProfileEncoder
    seed_everything(train_config.seed+101,train_config.threads)
    encoder_kwargs=dict(attention_layers=train_config.group_attention_layers,
                        attention_heads=train_config.attention_heads)
    # Production uses the full, activation-checkpointed encoder. A same-API
    # encoder can be injected for numerical interface tests without altering
    # the execution protocol or the covariance loss's global batch size.
    import inspect
    if "profile_chunk_size" in inspect.signature(encoder_class).parameters:
        encoder_kwargs["profile_chunk_size"]=config.profile_chunk_size
    encoder=encoder_class(kwargs["feature_groups"],train_config.hidden_dim,**encoder_kwargs)
    first=ConditionalJEPA(encoder,condition_dim=kwargs["condition_dim"],
                          reference_dim=kwargs["reference_dim"],chemical_dim=kwargs["chemical_dim"],
                          hidden_dim=train_config.hidden_dim,ema_decay=config.ema_reference_decay,
                          alignment_weight=25.,variance_weight=25.,covariance_weight=1.,
                          use_chemistry=train_config.use_chemistry,use_references=train_config.use_references,
                          use_library=train_config.use_library).to(train_config.device)
    second=copy.deepcopy(first)
    left,right=first.state_dict(),second.state_dict()
    if left.keys()!=right.keys() or any(not torch.equal(left[key],right[key]) for key in left):
        raise AssertionError("Diagnostic arms did not receive identical initial tensors")
    return dict(zip(ARM_NAMES,(first,second)))


def _diagnostic_indices(dataset,splits,config,seed):
    train=np.asarray(splits["train"],dtype=int)
    validation=np.asarray(splits["validation"],dtype=int)
    n=min(config.diagnostic_compounds,len(train),len(validation))
    rng=np.random.default_rng(seed+39019)
    # Selection uses row identity and a fixed seed, never measured outcomes.
    selected_train=np.sort(rng.permutation(train)[:n])
    selected_validation=np.sort(validation if len(validation)==n else rng.permutation(validation)[:n])
    return selected_train,selected_validation


def run_diagnostics(normalized_dataset,splits,train_config,kwargs,directory,*,
                    diagnostic_config=None,resume=False,encoder_class=None,
                    evaluate_fn=None,gradient_fn=None):
    """Run only paired JEPA diagnostics on the supplied normalized DEV split.

Preprocessing and library fitting must already be training-only. Latest states
are atomically committed after each complete arm-epoch, permitting interrupted
epochs to replay with the same episodes. Milestones export raw teacher weights;
no lowest-total-loss checkpoint is automatically selected.
"""
    config=(diagnostic_config or DiagnosticConfig()).validate()
    train_config.validate()
    if train_config.device!="cpu":
        raise ValueError("This paired diagnostic implementation uses the declared CPU path")
    assert_disjoint_splits(normalized_dataset.ids,splits)
    if not all((train_config.use_chemistry,train_config.use_references,train_config.use_library)):
        raise ValueError("This diagnostic retains chemistry, references and library context")
    if evaluate_fn is None or gradient_fn is None:
        from .jepa_diagnostic_metrics import evaluate_jepa,component_gradient_diagnostics
        evaluate_fn=evaluate_fn or evaluate_jepa
        gradient_fn=gradient_fn or component_gradient_diagnostics
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    initial_path=directory/"initial_state.pt"
    if initial_path.exists() and not resume:
        raise FileExistsError("A diagnostic initialization already exists; use its explicit resume path")
    _status(directory,"RUNNING","fresh_initialization_and_exact_pairing",epoch=0)
    learners=_make_learners(train_config,kwargs,config,encoder_class)
    if resume:
        initial=torch.load(initial_path,map_location="cpu",weights_only=True)
        for model in learners.values():
            current=model.state_dict()
            if current.keys()!=initial.keys() or any(not torch.equal(current[key],initial[key]) for key in current):
                raise ValueError("Resume fresh initialization does not match this diagnostic run")
    else:
        _atomic_torch(initial_path,next(iter(learners.values())).state_dict())
    anchor_model=next(iter(learners.values()))
    anchor=(copy.deepcopy(anchor_model.teacher_encoder).eval().requires_grad_(False),
            copy.deepcopy(anchor_model.teacher_projector).eval().requires_grad_(False))
    train=np.asarray(splits["train"],dtype=int)
    diagnostic_train,diagnostic_validation=_diagnostic_indices(normalized_dataset,splits,config,train_config.seed)
    gradient_indices=np.sort(np.random.default_rng(train_config.seed+98101).permutation(train)[:min(config.batch_size,len(train))])
    diagnostic_ids={"train":normalized_dataset.ids[diagnostic_train].tolist(),
                    "validation":normalized_dataset.ids[diagnostic_validation].tolist()}
    manifest=dict(schema_version=1,purpose="PAIRED_JEPA_OPTIMIZATION_DIAGNOSTIC_NOT_FORMAL_EXPERIMENT",
                  train_config=asdict(train_config),diagnostic_config=asdict(config),model_config=kwargs,
                  train_ids=normalized_dataset.ids[train].tolist(),
                  validation_ids=normalized_dataset.ids[splits["validation"]].tolist(),diagnostic_ids=diagnostic_ids,
                  gradient_diagnostic_ids=normalized_dataset.ids[gradient_indices].tolist(),
                  gradient_diagnostic_global_batch_size=len(gradient_indices),
                  feature_names=normalized_dataset.feature_names.tolist(),initialization_seed=train_config.seed+101,
                  normalization_metadata=normalized_dataset.metadata.get("train_scaler_applied"),
                  initial_tensors_equal=True,existing_dev_archive_loaded=True,
                  calibration_outcomes_used=False,evaluation_outcomes_used=False)
    # JSON canonicalization makes tuple/list serialization neutral on resume.
    manifest=json.loads(json.dumps(manifest,allow_nan=False))
    manifest_path=directory/"training_diagnostic_manifest.json"
    if resume and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError("Resume diagnostic configuration, feature geometry or source IDs changed")
    if not resume:
        _atomic_json(manifest_path,manifest)
    optimizers={name:torch.optim.AdamW([p for p in learner.parameters() if p.requires_grad],
                lr=config.base_lrs[index],weight_decay=config.weight_decay)
                for index,(name,learner) in enumerate(learners.items())}
    completed={name:0 for name in learners}
    missing_latest=[]
    for name in learners:
        arm_dir=directory/"arms"/name;arm_dir.mkdir(parents=True,exist_ok=True)
        latest=arm_dir/"latest.pt"
        if resume and latest.exists():
            saved=torch.load(latest,map_location="cpu",weights_only=True)
            if saved["manifest"]!=manifest or saved["arm"]!=name:
                raise ValueError("Resume checkpoint does not belong to this diagnostic arm")
            learners[name].load_state_dict(saved["state_dict"],strict=True)
            optimizers[name].load_state_dict(saved["optimizer"])
            learners[name].ema_decay=saved["ema_decay"]
            completed[name]=int(saved["epoch"])
        elif resume:
            missing_latest.append(name)
    probability_config=replace(train_config,batch_size=config.batch_size)
    batch_count=math.ceil(len(train)/config.batch_size)
    total_steps=config.epochs*batch_count
    began=time.monotonic()

    def diagnose(name,epoch):
        learner=learners[name]
        _status(directory,"RUNNING","detailed_diagnostics",epoch=epoch,arm=name,completed_epochs=completed)
        modes=[(module,module.training) for module in learner.modules()]
        try:
            with preserve_randomness():
                torch.manual_seed(train_config.seed+47017+epoch)
                _status(directory,"RUNNING","geometry_train_diagnostics",epoch=epoch,arm=name,completed_epochs=completed)
                train_metrics=evaluate_fn(learner,normalized_dataset,diagnostic_train,probability_config,
                                          chunk_size=config.eval_chunk_size,anchor=anchor)
                _status(directory,"RUNNING","geometry_validation_diagnostics",epoch=epoch,arm=name,completed_epochs=completed)
                validation_metrics=evaluate_fn(learner,normalized_dataset,diagnostic_validation,probability_config,
                                               chunk_size=config.eval_chunk_size,anchor=anchor)
                # A fixed decision-time state supplies component-gradient
                # diagnostics; it does not become another optimizer update.
                learner.train()
                _status(directory,"RUNNING","gradient_diagnostics",epoch=epoch,arm=name,completed_epochs=completed)
                gx,gy,gm=fixed_batch(normalized_dataset,gradient_indices,probability_config)
                gradients=gradient_fn(learner,{"inputs":gx,"target_y":gy,"target_mask":gm})
        finally:
            for module,mode in modes:
                module.training=mode
        return _record(directory/"diagnostics.jsonl","diagnostic",arm=name,epoch=epoch,
                       compounds_per_split=len(diagnostic_train),train=train_metrics,
                       validation=validation_metrics,gradient_compounds=len(gradient_indices),gradient_components=gradients)

    def checkpoint(name,epoch,milestone):
        learner=learners[name]
        saved=dict(schema_version=1,arm=name,epoch=epoch,manifest=manifest,
                   state_dict=learner.state_dict(),optimizer=optimizers[name].state_dict(),
                   encoder_state_dict=learner.teacher_encoder.state_dict(),
                   raw_teacher_encoder=True,ema_decay=learner.ema_decay,
                   torch_rng=torch.get_rng_state(),
                   random_stream_protocol="epoch_seeded_numpy_and_minibatch_torch",
                   next_episode_seed=train_config.seed+17+epoch*1000003,
                   next_global_step=epoch*batch_count,total_steps=total_steps,
                   checkpoint_selection="none_diagnostic_milestones_only")
        arm_dir=directory/"arms"/name
        _atomic_torch(arm_dir/"latest.pt",saved)
        if milestone:
            _atomic_torch(arm_dir/f"epoch_{epoch:03d}.pt",saved)

    try:
        _status(directory,"RUNNING","initialized",epoch=0,completed_epochs=completed)
        for name in learners:
            if not resume or name in missing_latest:
                diagnose(name,0);checkpoint(name,0,True)
        for epoch in range(min(completed.values())+1,config.epochs+1):
            for arm_index,(name,learner) in enumerate(learners.items()):
                if completed[name]>=epoch:
                    continue
                _status(directory,"RUNNING","training",epoch=epoch,arm=name,completed_epochs=completed)
                epoch_began=time.monotonic()
                learner.train();optimizer=optimizers[name]
                episode_rng=np.random.default_rng(train_config.seed+17+(epoch-1)*1000003)
                permutation=episode_rng.permutation(train)
                sums={};count=0;lrs=[];betas=[];sizes=[];norms=[]
                for batch_number,start in enumerate(range(0,len(train),config.batch_size)):
                    ix=permutation[start:start+config.batch_size]
                    item=random_training_batch(normalized_dataset,ix,episode_rng,probability_config)
                    lr=cosine_learning_rate(config.base_lrs[arm_index],config.base_lrs[arm_index]*config.min_lr_ratio,
                                            (epoch-1)*batch_count+batch_number,total_steps)
                    for group in optimizer.param_groups:group["lr"]=lr
                    beta=exposure_matched_ema(len(ix),config.ema_reference_decay,config.ema_reference_batch)
                    learner.ema_decay=beta
                    optimizer.zero_grad(set_to_none=True)
                    with _training_stream(train_config.seed,epoch-1,batch_number):
                        result=learner.loss(item["inputs"],item["target_y"],item["target_mask"])
                        if not torch.isfinite(result["loss"]):
                            raise FloatingPointError("Nonfinite JEPA diagnostic loss")
                        result["loss"].backward()
                    norm=torch.nn.utils.clip_grad_norm_(learner.parameters(),config.gradient_clip,error_if_nonfinite=True)
                    optimizer.step();learner.update_teacher()
                    norms.append(float(norm))
                    for key,value in result.items():
                        if torch.is_tensor(value) and value.numel()==1:
                            sums[key]=sums.get(key,0.)+float(value.detach())*len(ix)
                    count+=len(ix);lrs.append(lr);betas.append(beta);sizes.append(len(ix))
                    del result,item
                _record(directory/"training.jsonl","epoch",arm=name,epoch=epoch,
                        **{key:value/count for key,value in sums.items()},compounds=count,
                        lr_first=lrs[0],lr_last=lrs[-1],learning_rate_per_batch=lrs,
                        ema_decay_per_batch=betas,batch_compounds=sizes,
                        gradient_norm_max_before_clip=max(norms),gradient_norm_mean_before_clip=float(np.mean(norms)),
                        clipped_batch_fraction=float(np.mean(np.asarray(norms)>config.gradient_clip)),
                        optimizer_steps_this_epoch=len(sizes),optimizer_steps_total=epoch*batch_count,
                        epoch_duration_seconds=time.monotonic()-epoch_began,elapsed_seconds=time.monotonic()-began)
                milestone=epoch==1 or epoch%config.report_every==0 or epoch==config.epochs
                if milestone:diagnose(name,epoch)
                checkpoint(name,epoch,milestone)
                completed[name]=epoch
                _status(directory,"RUNNING","arm_epoch_complete",epoch=epoch,arm=name,completed_epochs=completed)
        summary=dict(state="DIAGNOSTIC_COMPLETE",epochs_by_arm=completed,arms=list(learners),
                     diagnostic_ids=diagnostic_ids,checkpoint_selection="none_diagnostic_milestones_only",
                     world_model_training_started=False,existing_dev_archive_loaded=True,
                     calibration_outcomes_used=False,evaluation_outcomes_used=False)
        _atomic_json(directory/"diagnostic_summary.json",summary)
        _status(directory,"DIAGNOSTIC_COMPLETE","complete",epoch=config.epochs,completed_epochs=completed)
        return summary
    except BaseException as error:
        _status(directory,"FAILED","interrupted_or_error",completed_epochs=completed,
                error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
