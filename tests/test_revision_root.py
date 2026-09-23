"""Regression checks for audit failures and complete-run orchestration."""
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from opal2.cli import arm_config
from opal2.config import TrainConfig
from opal2.export import export_source
from opal2.provenance import bind_fitting_provenance, assert_evaluation_provenance
from opal2.experiments import _fit_partition, _folds
from test_training_integration import schema_fixture
from opal2.data import TrainScaler


def test_checkpoint_validation_cannot_be_relabelled_as_evaluation():
    ds = schema_fixture(d=8,n=12)
    scaler = TrainScaler.fit(ds,np.arange(6))
    model = torch.nn.Linear(1,1)
    bind_fitting_provenance(model,ds.ids[:6],ds.ids[6:8])
    honest = {"train":np.arange(6),"validation":np.array([6,7]),
              "calibration":np.array([8,9]),"evaluation":np.array([10,11])}
    assert_evaluation_provenance(model,scaler,ds,honest)
    swapped = dict(honest,validation=np.array([10,11]),evaluation=np.array([6,7]))
    with pytest.raises(ValueError,match="Checkpoint"):
        assert_evaluation_provenance(model,scaler,ds,swapped)


def test_full_default_and_freezing_controls_are_explicit():
    config=TrainConfig().validate()
    assert (config.hidden_dim,config.latent_rank,config.residual_rank)==(256,32,8)
    assert config.samples==2000 and config.mc_chunk_size==32
    assert arm_config(config,"D").resolved_encoder_policy=="jepa_frozen"
    assert arm_config(config,"FR").resolved_encoder_policy=="frozen_random"
    assert arm_config(config,"JF").resolved_encoder_policy=="jepa_finetune"
    assert not arm_config(config,"NO_CHEM").use_chemistry
    with pytest.raises(ValueError):
        replace(config,samples=1.2).validate()
    with pytest.raises(ValueError,match="real pretrained"):
        replace(config,encoder_policy="pretrained_frozen").validate()


def test_nested_partition_never_uses_outer_outcomes_for_fitting():
    ds=schema_fixture(d=5,n=60)
    for pool,heldout in _folds(np.arange(60),3,17):
        inner,split=_fit_partition(ds,pool,heldout,19)
        assert set(inner.ids[split["evaluation"]])==set(ds.ids[heldout])
        used=set(inner.ids[np.r_[split["train"],split["validation"],split["calibration"]]])
        assert used==set(ds.ids[pool])
        assert not used&set(ds.ids[heldout])


def test_shuffled_groups_preserve_dimensions_sizes_and_are_reproducible():
    from opal2.training import model_kwargs
    ds=schema_fixture(d=24,n=12)
    config=arm_config(TrainConfig(),"SHUFFLED_GROUPS")
    first=model_kwargs(ds,config)["feature_groups"]
    second=model_kwargs(ds,config)["feature_groups"]
    assert first==second
    assert [len(x) for x in first.values()]==[len(x) for x in ds.feature_groups.values()]
    assert sorted(sum(first.values(),[]))==list(range(24))
    assert first!=ds.feature_groups


def test_export_includes_real_prediction_filename_and_new_diagnostics(tmp_path):
    root=tmp_path/"project"
    (root/"runs"/"D").mkdir(parents=True)
    for name in ("pyproject.toml","requirements-tested.txt"):
        (root/name).write_text("test fixture\n")
    for name in ("evaluation_predictions.tsv","norm_diagnostics.tsv","mc_precision.tsv","workflow.json","decision_workflow.json"):
        (root/"runs"/"D"/name).write_text("test fixture\n")
    target=tmp_path/"source.zip"
    export_source(root,target)
    with ZipFile(target) as archive:
        names=archive.namelist()
        assert "project/runs/D/evaluation_predictions.tsv" in names
        assert "project/runs/D/workflow.json" in names
        assert "project/runs/D/decision_workflow.json" in names


def test_selective_probe_cli_forwards_declared_setup_and_cost_settings(tmp_path,monkeypatch):
    import json
    import opal2.cli as cli
    import opal2.training as training
    import opal2.probe as probe
    calls=[]
    config=TrainConfig()
    monkeypatch.setattr(cli,"get_data",lambda args:("numerical fixture",{"evaluation":np.array([0,1])}))
    monkeypatch.setattr(training,"load_model",lambda path:(None,None,config,{}))
    monkeypatch.setattr(training,"seed_everything",lambda *a:None)
    def capture(*a,**kw):
        calls.append(kw)
        return {"mode":"CLI_WIRING_NUMERICAL_TEST"}
    monkeypatch.setattr(probe,"run_selective_model_probe",capture)
    settings={"target_costs":[.01,.02,0],"setup_memberships":{"batch_b":[[True,False,False],[True,False,False]]},
              "setup_costs":{"batch_b":.03},"cost_budget":.06,"max_total_null_fraction":.35,
              "risk_penalty":.02,"missing_outcome":"worst"}
    path=tmp_path/"planning.json";path.write_text(json.dumps(settings))
    cli.main(["selective-probe","--data","fixture","--model","fixture","--budget","2",
              "--planning-settings",str(path),"--output",str(tmp_path/"result.json")])
    assert all(calls[0][key]==value for key,value in settings.items())
    assert calls[0]["selection_samples"]==2000
