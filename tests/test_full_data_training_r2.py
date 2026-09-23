"""Complete model/data wiring on explicitly numerical, non-biological fixtures."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from test_data import records
from test_data_fidelity_r2 import panels
from opal2.config import TrainConfig
from opal2.data import (LibraryBank, attach_library_context, fit_library_context,
                       load_dataset, save_dataset, TrainScaler)
from opal2.training import fit_model, load_model, fixed_batch, random_training_batch
from opal2.cli import main
from opal2.splits import load_split


def test_complete_two_stage_training_compact_controls_bank_and_reload(tmp_path):
    """Two actual JEPA + two actual variational epochs; not performance evidence."""
    dataset = panels(records(n=20,w=4))
    splits = {"train":np.arange(12),"validation":np.arange(12,15),
              "calibration":np.arange(15,17),"evaluation":np.arange(17,20)}
    config = TrainConfig(hidden_dim=16,latent_rank=2,residual_rank=2,
                         group_attention_layers=2,attention_heads=2,
                         epochs=2,jepa_epochs=2,batch_size=4,patience=2,
                         threads=1,samples=8,mc_chunk_size=4,use_jepa=True,use_library=True)
    model,scaler = fit_model(dataset,splits,config,tmp_path/"full_model")
    loaded,restored,loaded_config,payload = load_model(tmp_path/"full_model")
    assert loaded_config == config
    assert loaded.library_bank.ids.tolist() == dataset.ids[splits["train"]].tolist()
    assert restored.template_panel_ids == scaler.template_panel_ids
    assert len(restored.template_panel_ids) == 4
    normalized = attach_library_context(restored.transform(dataset),loaded.library_bank)
    inputs,_,_ = fixed_batch(normalized,splits["evaluation"],config)
    assert inputs["panel_catalog_y"].shape == (4,4)
    assert inputs["context_panel_mask"].sum() == 6
    assert inputs["context_panel_template_mask"].sum() == 6
    assert inputs["library_index"].shape == (3,12)
    with torch.no_grad():
        a,b = model(inputs),loaded(inputs)
    assert torch.equal(a.mean,b.mean)
    assert torch.equal(a.marginal_variance,b.marginal_variance)
    log = [json.loads(x) for x in (tmp_path/"full_model"/"training.jsonl").read_text().splitlines()]
    assert sum(x["event"] == "jepa_epoch" for x in log) == 2
    assert sum(x["event"] == "world_model_epoch" for x in log) == 2
    assert payload["validation_ids"] == dataset.ids[splits["validation"]].tolist()
    assert all(torch.isfinite(torch.tensor(x["train_objective"])) for x in log if x["event"] == "world_model_epoch")
    bank = LibraryBank.load(tmp_path/"full_model"/"library_context.npz")
    bank.ids[-1] = dataset.ids[17]
    bank.save(tmp_path/"full_model"/"library_context.npz")
    with pytest.raises(ValueError,match="non-training"):
        load_model(tmp_path/"full_model")


@pytest.mark.parametrize("flag",["use_chemistry","use_references","use_library"])
def test_ablation_information_mask_applies_to_both_training_and_fixed_batches(flag):
    dataset = panels(records(n=12,w=4))
    scaler = TrainScaler.fit(dataset,np.arange(8))
    norm = scaler.transform(dataset)
    norm = attach_library_context(norm,fit_library_context(norm,np.arange(8)))
    config = replace(TrainConfig(),**{flag:False})
    batch = random_training_batch(norm,np.array([0,1,2]),np.random.default_rng(21),config)["inputs"]
    fixed,_,_ = fixed_batch(norm,np.array([0,1,2]),config)
    for value in (batch,fixed):
        if flag == "use_chemistry":
            assert not value["chem_mask"].any() and not value["chem"].any()
        elif flag == "use_references":
            assert not value["context_reference_mask"].any()
            assert not value["target_reference_mask"].any()
            assert not value["context_panel_mask"].any()
            assert not value["target_panel_mask"].any()
        else:
            assert not value["library_mask"].any()
            assert not value["library_y"].any()
            assert not value["library_global_mean"].any()


def test_prepare_source_holdout_cli_roundtrip_actual_four_role_schema(tmp_path):
    dataset = records(n=16,w=4)
    source_id = np.repeat(np.arange(4),4)
    dataset.groups[:,:,0] = source_id[:,None]
    sources = np.repeat(np.array(["source_a","source_b","source_c","source_d"])[source_id,None],4,axis=1)
    save_dataset(dataset,tmp_path/"supplied_measurements")
    partition = {"well_sources":sources.tolist(),"train_sources":["source_a"],"validation_sources":["source_b"],
                 "calibration_sources":["source_c"],"evaluation_sources":["source_d"]}
    (tmp_path/"partition.json").write_text(json.dumps(partition))
    main(["prepare-source-holdout","--dataset",str(tmp_path/"supplied_measurements.npz"),
          "--partition",str(tmp_path/"partition.json"),"--output",str(tmp_path/"prepared")])
    output = load_dataset(tmp_path/"prepared"/"measurements.npz")
    splits,scope = load_split(tmp_path/"prepared"/"splits.json",output.ids)
    np.testing.assert_array_equal(output.Y,dataset.Y)
    assert output.ids.tolist() == dataset.ids.tolist()
    assert {name:len(ix) for name,ix in splits.items()} == {"train":4,"validation":4,"calibration":4,"evaluation":4}
    assert scope == "DECLARED_SOURCE_AND_COMPOUND_HOLDOUT_FOUR_ROLE_TASK"
    for name,index in zip(("train","validation","calibration","evaluation"),range(4)):
        assert set(output.groups[splits[name],:,0].ravel()) == {index}


def test_nested_manifest_and_inner_selection_precede_outer_use(tmp_path,monkeypatch):
    """Control-flow audit with spies, separate from the actual training test."""
    import opal2.experiments as experiments
    dataset = records(n=60,w=4)
    cfg = TrainConfig(hidden_dim=16,latent_rank=2,residual_rank=2,
                      attention_heads=2,epochs=2,jepa_epochs=2,threads=1,samples=8)
    output = tmp_path/"nested"
    outer = list(experiments._folds(np.arange(len(dataset)),3,44))
    training_calls = []

    def fit_spy(ds,split,config,path):
        path = type(output)(path)
        manifest = json.loads((output/"fold_manifest.json").read_text())
        assert set(manifest["configurations"]) == {"A","B"}
        outer_number = int(next(x for x in path.parts if x.startswith("outer_")).split("_")[1])
        outer_ids = set(dataset.ids[outer[outer_number][1]])
        if "inner_" in str(path):
            assert not outer_ids.intersection(ds.ids)
        else:
            selection = json.loads((path.parent/"selection.json").read_text())
            assert selection["chosen"] == "A"
            assert set(selection["outer_ids_not_used"]) == outer_ids
            assert not outer_ids.intersection(ds.ids[np.r_[split["train"],split["validation"],split["calibration"]]])
        path.mkdir(parents=True,exist_ok=False)
        training_calls.append(path)
        return None,None

    monkeypatch.setattr(experiments,"fit_model",fit_spy)
    monkeypatch.setattr(experiments,"load_model",lambda path:(None,None,cfg,{}))
    monkeypatch.setattr(experiments,"_evaluate_frozen",lambda *args:([{"population_mean_net_gain":0.}],[],{}))
    reports = experiments.nested_development_cv(dataset,{"A":cfg,"B":replace(cfg,kernel_mode="mlp")},
                                              output,outer_folds=3,inner_folds=2,seed=44)
    assert len(training_calls) == 3*(2*2+1)
    assert all(row["selected"] == "A" for row in reports)
    assert not json.loads((output/"cross_validation.json").read_text())["formal_certificate"]
