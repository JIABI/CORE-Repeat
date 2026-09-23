"""Paired optimization protocol tests using labeled numerical fixtures only."""
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from opal2.config import TrainConfig
from opal2.data import TrainScaler,fit_library_context,attach_library_context
from opal2.jepa_diagnostic_training import (DiagnosticConfig,cosine_learning_rate,
    exposure_matched_ema,preserve_randomness,_make_learners,run_diagnostics,ARM_NAMES)
from opal2.model import GroupedProfileEncoder
from opal2.training import model_kwargs
from test_training_integration import schema_fixture


def test_cosine_is_monotone_and_reaches_both_actual_step_endpoints():
    for base in (3e-4,1e-4):
        sequence=np.array([cosine_learning_rate(base,base*.01,i,180) for i in range(180)])
        assert sequence[0]==base
        assert sequence[-1]==base*.01
        assert (np.diff(sequence)<=0).all()
    with pytest.raises(ValueError):cosine_learning_rate(3e-4,3e-6,180,180)


def test_ema_matches_sample_exposure_not_number_of_optimizer_updates():
    assert exposure_matched_ema(128)==pytest.approx(.96059601)
    assert exposure_matched_ema(32)==.99
    actual=np.prod([exposure_matched_ema(n) for n in (128,128,127)])
    assert actual==pytest.approx(.99**(383/32))
    assert exposure_matched_ema(127)>exposure_matched_ema(128)


def test_diagnostic_randomness_scope_restores_torch_numpy_and_python_even_on_error():
    torch.manual_seed(12);np.random.seed(13);random.seed(14)
    ts=torch.get_rng_state().clone();ns=np.random.get_state();ps=random.getstate()
    with pytest.raises(RuntimeError):
        with preserve_randomness():
            torch.rand(12);np.random.rand(17);random.random()
            raise RuntimeError("fixture interruption")
    assert torch.equal(ts,torch.get_rng_state())
    assert ns[0]==np.random.get_state()[0] and np.array_equal(ns[1],np.random.get_state()[1])
    assert ns[2:]==np.random.get_state()[2:]
    assert ps==random.getstate()


def _fixture():
    dataset=schema_fixture(d=9,n=12)
    splits={"train":np.arange(8),"validation":np.array([8,9]),
            "calibration":np.array([10]),"evaluation":np.array([11])}
    normalized=TrainScaler.fit(dataset,splits["train"]).transform(dataset)
    normalized=attach_library_context(normalized,fit_library_context(normalized,splits["train"]))
    # If the runner accesses either protected outcome it cannot obtain a
    # finite embedding. Neither partition is used for this diagnostic.
    normalized.Y[10:]=np.nan
    cfg=TrainConfig(seed=1947,hidden_dim=16,latent_rank=2,residual_rank=1,threads=1)
    diag=DiagnosticConfig(epochs=2,batch_size=4,report_every=1,diagnostic_compounds=2,
                          eval_chunk_size=1,profile_chunk_size=1)
    return normalized,splits,cfg,diag,model_kwargs(normalized,cfg)


def test_fresh_arms_have_exactly_equal_parameters_and_complete_jepa_loss():
    _,_,cfg,diag,kwargs=_fixture()
    arms=_make_learners(cfg,kwargs,diag,GroupedProfileEncoder)
    a,b=arms.values()
    assert all(torch.equal(a.state_dict()[key],b.state_dict()[key]) for key in a.state_dict())
    assert (a.alignment_weight,a.variance_weight,a.covariance_weight)==(25.,25.,1.)
    assert a.use_library and a.use_references and a.use_chemistry
    assert a.student_encoder.hidden_dim==cfg.hidden_dim


def test_interleaved_training_diagnostic_rng_and_resume(tmp_path):
    dataset,splits,cfg,diag,kwargs=_fixture()
    calls=[]
    def evaluate(model,ds,indices,config,chunk_size=16,anchor=None):
        assert set(indices).issubset(set(splits["train"])|set(splits["validation"]))
        assert len(indices)==2 and chunk_size==1 and anchor is not None
        calls.append(tuple(indices))
        torch.rand(11);np.random.rand(11);random.random()
        return {"fixture":True,"count":len(indices)}
    def gradients(model,item):
        # Geometry uses train2 versus val2, but gradient covariance must use
        # the FULL configured batch4, not the profile/evaluation chunk1.
        assert item["target_y"].shape[0]==4
        torch.rand(15)
        return {"fixture":True,"compound_count":4}
    result=run_diagnostics(dataset,splits,cfg,kwargs,tmp_path/"consume_rng",diagnostic_config=diag,
                           encoder_class=GroupedProfileEncoder,evaluate_fn=evaluate,gradient_fn=gradients)
    assert result["state"]=="DIAGNOSTIC_COMPLETE"
    records=[json.loads(x) for x in (tmp_path/"consume_rng"/"training.jsonl").read_text().splitlines()]
    assert [(x["epoch"],x["arm"]) for x in records]==[(1,ARM_NAMES[0]),(1,ARM_NAMES[1]),(2,ARM_NAMES[0]),(2,ARM_NAMES[1])]
    assert all(x["batch_compounds"]==[4,4] and x["optimizer_steps_this_epoch"]==2 for x in records)
    assert all(0<=x["clipped_batch_fraction"]<=1 for x in records)
    assert all(x["gradient_norm_mean_before_clip"]<=x["gradient_norm_max_before_clip"] for x in records)
    manifest=json.loads((tmp_path/"consume_rng"/"training_diagnostic_manifest.json").read_text())
    assert manifest["gradient_diagnostic_global_batch_size"]==4
    assert len(manifest["gradient_diagnostic_ids"])==4
    def quiet_eval(*args,**kwargs):return {"fixture":True}
    def quiet_grad(*args,**kwargs):return {"fixture":True}
    run_diagnostics(dataset,splits,cfg,kwargs,tmp_path/"quiet_rng",diagnostic_config=diag,
                    encoder_class=GroupedProfileEncoder,evaluate_fn=quiet_eval,gradient_fn=quiet_grad)
    for arm in ARM_NAMES:
        one=torch.load(tmp_path/"consume_rng"/"arms"/arm/"latest.pt",weights_only=True)
        two=torch.load(tmp_path/"quiet_rng"/"arms"/arm/"latest.pt",weights_only=True)
        assert all(torch.equal(one["state_dict"][key],two["state_dict"][key]) for key in one["state_dict"])
        assert one["checkpoint_selection"]=="none_diagnostic_milestones_only"
        assert one["epoch"]==2 and one["raw_teacher_encoder"]
        assert (tmp_path/"consume_rng"/"arms"/arm/"epoch_000.pt").exists()
        assert (tmp_path/"consume_rng"/"arms"/arm/"epoch_001.pt").exists()
    count=len(calls)
    resumed=run_diagnostics(dataset,splits,cfg,kwargs,tmp_path/"consume_rng",diagnostic_config=diag,
                            resume=True,encoder_class=GroupedProfileEncoder,evaluate_fn=evaluate,gradient_fn=gradients)
    assert resumed["state"]=="DIAGNOSTIC_COMPLETE" and len(calls)==count
    status=json.loads((tmp_path/"consume_rng"/"status.json").read_text())
    assert status["state"]=="DIAGNOSTIC_COMPLETE"


def test_failure_status_contains_traceback_and_uncommitted_epoch_is_not_claimed(tmp_path):
    dataset,splits,cfg,diag,kwargs=_fixture()
    def fail(*args,**kwargs):raise RuntimeError("fixture diagnostic failure")
    with pytest.raises(RuntimeError,match="fixture diagnostic failure"):
        run_diagnostics(dataset,splits,cfg,kwargs,tmp_path,diagnostic_config=diag,
                        encoder_class=GroupedProfileEncoder,evaluate_fn=fail,gradient_fn=fail)
    status=json.loads((tmp_path/"status.json").read_text())
    assert status["state"]=="FAILED" and "RuntimeError" in status["traceback"]
    assert status["completed_epochs"]==dict.fromkeys(ARM_NAMES,0)
