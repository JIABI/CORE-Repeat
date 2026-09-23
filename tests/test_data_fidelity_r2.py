"""Identity, availability, catalog and training-only context regression tests."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_data import records
from opal2.data import (TrainScaler, LibraryBank, attach_library_context, fit_library_context,
                       make_inference_batch, make_episode, collate_episodes, save_dataset, load_dataset)
from opal2.splits import materialize_source_partitions, merge_compound_disjoint_partitions


def panels(ds):
    m,d = 4,ds.Y.shape[-1]
    pi = np.full((*ds.Y.shape[:2],3,2),-1,int)
    for row in range(len(ds)):
        pi[row,:,2] = (row%2*2,row%2*2+1)
    return replace(ds,panel_y=np.arange(m*d).reshape(m,d)/10,
                   panel_ids=np.array([f"panel{i}" for i in range(m)]),
                   panel_identity=np.array(["positive","DMSO","positive","DMSO"]),
                   panel_members=np.array([[f"control{i}"] for i in range(m)]),
                   panel_groups=np.array([[i//2,0,i//2] for i in range(m)]),panel_index=pi,
                   panel_template=None,panel_template_mask=None)


def test_identity_rejects_duplicate_real_well_and_fractional_indices():
    ds = records()
    ids = ds.well_ids.copy()
    ids[0,1] = ids[0,0]
    with pytest.raises(ValueError,match="globally unique"):
        replace(ds,well_ids=ids)
    with pytest.raises(ValueError,match="physical well_ids"):
        replace(ds,well_ids=None,metadata={})
    fractional = ds.reference_mask.astype(float)
    fractional[0,0,1] = .3
    with pytest.raises(ValueError,match="binary masks"):
        replace(ds,reference_mask=fractional)
    for compounds,context,target in (([.5],(0,),(1,)),([0],(.1,),(1,)),([0],(0,),(1.9,))):
        with pytest.raises(ValueError,match="exact integers"):
            make_inference_batch(ds,compounds,context,target)
    ds.well_ids[0,1] = ds.well_ids[0,0]
    with pytest.raises(ValueError,match="physical well identity"):
        make_inference_batch(ds,[0],(0,),(1,))


def test_zero_context_missing_chemistry_and_future_cell_counts():
    ds = records()
    ds.chem_mask[0] = False
    ds.chem[0] = np.nan
    ds.n_cells[:] = 100
    ds.n_cells_mask[:] = True
    batch = make_inference_batch(ds,[0],(),(0,1))
    assert batch["context_y"].shape == (1,0,4)
    assert not batch["chem"].any() and not batch["chem_mask"].any()
    assert not batch["target_n_cells_mask"].any() and not batch["target_n_cells"].any()
    a = make_episode(ds,0,(),(0,1))
    b = make_episode(ds,1,(0,),(1,2))
    joined = collate_episodes([a,b])
    assert joined["inputs"]["context_mask"].tolist() == [[False],[True]]
    assert joined["inputs"]["chem_mask"].tolist() == [False,True]


def test_reference_identity_templates_are_training_only_and_leave_reference_out(tmp_path):
    ds = panels(records())
    scaler = TrainScaler.fit(ds,[0,2,4])
    poisoned = replace(ds,panel_y=ds.panel_y.copy())
    poisoned.panel_y[2:] += 1e6
    assert scaler.to_dict() == TrainScaler.fit(poisoned,[0,2,4]).to_dict()
    normalized = scaler.transform(ds)
    # Training has one observation of each reference identity. It cannot target
    # itself as a typical profile, but the other-source identity is anchored.
    assert normalized.panel_template_mask.tolist() == [False,False,True,True]
    assert np.allclose(normalized.panel_template[2],scaler.transform_y(ds.panel_y[0]))
    a = make_inference_batch(normalized,[1],(0,),(1,),reference_access="observed_only")
    assert a["context_panel_mask"].sum() == 2
    assert not a["target_panel_mask"].any()
    assert a["context_panel_template_mask"].sum() == 2
    save_dataset(ds,tmp_path/"panels")
    restored = load_dataset(tmp_path/"panels")
    np.testing.assert_array_equal(restored.panel_members,ds.panel_members)
    np.testing.assert_array_equal(restored.panel_y,ds.panel_y)


def test_library_only_train_initial_wells_excludes_self_and_target_poison(tmp_path):
    ds = records()
    scaler = TrainScaler.fit(ds,np.arange(8))
    scaled = scaler.transform(ds)
    bank = fit_library_context(scaled,np.arange(8),max_neighbors=4)
    bank.save(tmp_path/"library")
    bank = LibraryBank.load(tmp_path/"library")
    use = attach_library_context(scaled,bank)
    before = make_inference_batch(use,[0,9],(0,),(1,2))
    assert before["library_count"].tolist() == [7,8]
    assert before["library_y"].shape == (2,4,4)
    for row in range(2):
        np.testing.assert_array_equal(before["library_y"][row],bank.Y[before["library_index"][row]].astype(np.float32))
    assert not any(torch.equal(row,before["context_y"][0,0]) for row in before["library_y"][0])
    use.Y[:,1:] += 1e8
    after = make_inference_batch(use,[0,9],(0,),(1,2))
    assert all(torch.equal(before[key],after[key]) for key in before)
    empty = make_inference_batch(use,[9],(),(1,))
    use.Y[9] += 1e8
    empty2 = make_inference_batch(use,[9],(),(1,))
    assert all(torch.equal(empty[key],empty2[key]) for key in empty)


def test_source_materialization_removes_heldout_outcomes_and_reference_rows():
    ds = panels(records(n=12,w=8))
    sources = np.tile(np.repeat(["a","b","c","d"],2),(len(ds),1))
    ds.groups[:,:,0] = np.tile(np.repeat(np.arange(4),2),(len(ds),1))
    ds.panel_groups[:,0] = np.arange(4)
    ds.panel_index[:] = -1
    for j in range(8):
        ds.panel_index[:,j,2,0] = j//2
    kwargs = dict(train_sources=["a"],validation_sources=["b"],calibration_sources=["c"],evaluation_sources=["d"])
    with pytest.raises(ValueError,match="Compound overlap"):
        materialize_source_partitions(ds,sources,**kwargs)
    partition = materialize_source_partitions(ds,sources,allow_known_compounds=True,**kwargs)
    for i,name in enumerate(("train","validation","calibration","evaluation")):
        part = partition[name]
        assert part.Y.shape == (12,2,4)
        np.testing.assert_array_equal(part.Y,ds.Y[:,2*i:2*i+2])
        assert part.metadata["source_masks_materialized_before_pairing"]
        assert not part.panel_template_mask.any()


def test_reference_cannot_include_treatment_well():
    ds = panels(records())
    members = ds.panel_members.astype(object)
    members[0,0] = ds.well_ids[0,0]
    with pytest.raises(ValueError,match="disjoint"):
        replace(ds,panel_members=members)


def test_compact_catalog_collation_and_hidden_control_poison():
    ds = panels(records(w=4))
    ds.panel_index[:] = -1
    ds.panel_index[:,0,2] = (0,1)
    ds.panel_index[:,1:,2] = (2,3)
    scaled = TrainScaler.fit(ds,[0,1,2]).transform(ds)
    before = make_inference_batch(scaled,[4],(0,),(1,2))
    scaled.panel_y[2:] += 1e8
    after = make_inference_batch(scaled,[4],(0,),(1,2))
    assert all(torch.equal(before[k],after[k]) for k in before)
    episodes = [make_episode(scaled,4,(0,),(1,2)),make_episode(scaled,5,(1,),(2,3))]
    batch = collate_episodes(episodes)["inputs"]
    assert batch["panel_catalog_ids"].tolist() == [0,1,2,3]
    assert batch["context_panel_index"][0,0,2].tolist() == [0,1]
    assert batch["context_panel_index"][1,0,2].tolist() == [2,3]


def test_merge_real_source_partitions_does_not_rename_compounds():
    ds = records(n=12,w=4)
    sources = np.repeat(np.repeat(["a","b","c","d"],3)[:,None],4,axis=1)
    ds.groups[:,:,0] = np.repeat(np.repeat(np.arange(4),3)[:,None],4,axis=1)
    parts = materialize_source_partitions(ds,sources,train_sources=["a"],validation_sources=["b"],
                                         calibration_sources=["c"],evaluation_sources=["d"])
    merged,splits = merge_compound_disjoint_partitions(parts)
    assert merged.ids.tolist() == ds.ids.tolist()
    assert [len(splits[k]) for k in ("train","validation","calibration","evaluation")] == [3,3,3,3]
    np.testing.assert_array_equal(merged.Y,ds.Y)
    assert merged.metadata["evidence_scope"] == "SOURCE_AND_COMPOUND_DISJOINT_PORTABLE_DATA"
