"""Unit fixtures exercise data interfaces; they are not biological evidence."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from opal2.data import (MeasurementDataset, TrainScaler, EpisodeDataset, cellprofiler_groups,
                       collate_episodes, load_dataset, save_dataset, split_compounds,
                       make_episode, make_inference_batch, make_training_batch)
from opal2.source5 import morgan_fingerprints


def records(n=12, w=5):
    rng = np.random.default_rng(741)
    names = np.array(["Cells_AreaShape_Area", "Cells_AreaShape_Perimeter",
                      "Nuclei_Intensity_MeanIntensity_DNA", "Cytoplasm_Texture_Contrast_ER_3_00"])
    gi, gn = cellprofiler_groups(names)
    mask = np.ones((n, w, 3), bool)
    mask[..., 0] = False
    return MeasurementDataset(
        Y=rng.normal(size=(n, w, len(names))), ids=np.array([f"unit_{i}" for i in range(n)]),
        feature_names=names, feature_group_index=gi, feature_group_names=gn,
        cond=rng.normal(size=(n, w, 2)), reference=rng.normal(size=(n, w, 3, 6)),
        reference_mask=mask,
        groups=np.tile(np.stack([np.zeros(w), np.arange(w), np.arange(w)], -1)[None], (n, 1, 1)).astype(int),
        chem=rng.integers(0, 2, size=(n, 17)).astype(float), metadata={"fixture": True})


def test_feature_grouping_is_named_and_complete():
    ds = records()
    assert ds.feature_groups["Cells::AreaShape"] == [0, 1]
    assert sorted(sum(ds.feature_groups.values(), [])) == list(range(4))


def test_portable_roundtrip_without_pickle(tmp_path):
    ds = records()
    path, sidecar = save_dataset(ds, tmp_path / "fixture")
    result = load_dataset(path)
    assert result.feature_groups == ds.feature_groups
    for field in ("Y", "ids", "groups", "cond", "reference", "reference_mask", "chem", "well_ids"):
        np.testing.assert_array_equal(getattr(result, field), getattr(ds, field))
    assert json.loads(sidecar.read_text())["schema_version"] == 2
    with pytest.raises(FileExistsError):
        save_dataset(ds, path)


def test_scaler_uses_only_training_groups_and_inverts_mc(tmp_path):
    ds = records()
    train = np.arange(7)
    a = TrainScaler.fit(ds, train)
    poisoned = replace(ds, Y=ds.Y.copy(), cond=ds.cond.copy(), reference=ds.reference.copy())
    poisoned.Y[7:] += 1e8
    poisoned.cond[7:] += 1e9
    poisoned.reference[7:] += 1e10
    b = TrainScaler.fit(poisoned, train)
    assert a.to_dict() == b.to_dict()
    a.save(tmp_path / "scaler.json")
    restored = TrainScaler.load(tmp_path / "scaler.json")
    scaled = restored.transform(ds)
    np.testing.assert_allclose(restored.inverse_y(scaled.Y), ds.Y, atol=1e-12)
    mc = torch.tensor(np.stack([scaled.Y[:2]] * 3), dtype=torch.float64)
    np.testing.assert_allclose(restored.inverse_y(mc).numpy(), np.stack([ds.Y[:2]] * 3), atol=1e-12)
    with pytest.raises(ValueError, match="twice"):
        restored.transform(scaled)


def test_inference_never_contains_or_depends_on_target_outcomes():
    ds = records()
    before = make_inference_batch(ds, [0, 1], (0,), (1, 2, 3))
    ds.Y[:, 1:] = np.nan  # A future outcome need not exist at inference time.
    after = make_inference_batch(ds, [0, 1], (0,), (1, 2, 3))
    assert "target_y" not in after
    assert "Gamma" not in after
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_target_reference_is_hidden_even_when_source_matches_context():
    ds = records()
    before = make_inference_batch(ds, [0], (0,), (1, 2))
    assert not before["target_reference_mask"].any()
    assert not before["target_reference"].any()
    assert before["context_reference_mask"].sum() == 2
    ds.reference[:, 1:] += 123456789
    after = make_inference_batch(ds, [0], (0,), (1, 2))
    for key in before:
        assert torch.equal(before[key], after[key]), key
    declared = make_inference_batch(ds, [0], (0,), (1, 2), reference_access="all_declared")
    assert declared["target_reference_mask"].sum() == 4
    assert declared["target_reference"].abs().sum() > 0
    none = make_inference_batch(ds, [0], (0,), (1, 2), reference_access="none")
    assert not none["context_reference_mask"].any()


def test_missing_outcomes_and_padding_are_not_targets():
    ds = records(n=3)
    ds.Y[0, 4] = np.nan
    ds.observed_mask[0, 4] = False
    episode = make_episode(ds, 0, (0,), (1, 4))
    assert episode["target_mask"].tolist() == [True, False]
    assert torch.isfinite(episode["target_y"]).all()
    assert torch.equal(episode["target_y"][1], torch.zeros(4))
    other = make_episode(ds, 1, (0, 1), (2, 3, 4))
    batch = collate_episodes([episode, other])
    assert batch["inputs"]["context_y"].shape == (2, 2, 4)
    assert batch["target_y"].shape == (2, 3, 4)
    assert batch["inputs"]["context_mask"].tolist() == [[True, False], [True, True]]
    assert batch["target_mask"].tolist() == [[True, False, False], [True, True, True]]
    assert batch["inputs"]["target_group"][0, 2].tolist() == [-1, -1, -1]


def test_split_precedes_pairing_and_respects_coarser_groups():
    ds = records()
    group = np.repeat(["batch_A", "batch_B", "batch_C", "batch_D", "batch_E", "batch_F"], 2)
    allocation = split_compounds(ds, fractions=(.5, .25, .25), groups=group, seed=13)
    seen = set()
    for indices in allocation.values():
        local = set(group[indices])
        assert not seen & local
        seen |= local
        episodes = EpisodeDataset(ds, indices, context_sizes=(1, 2, 4))
        assert {e[0] for e in episodes.episodes} == set(indices)
        assert all(not set(e[1]) & set(e[2]) for e in episodes.episodes)
        assert {len(e[1]) for e in episodes.episodes} == {1, 2, 4}
    with pytest.raises(ValueError):
        split_compounds(ds, groups=np.repeat("one_source", len(ds)))


def test_episode_expansion_count_and_fixed_training_batch():
    ds = records(w=4)
    episodes = EpisodeDataset(ds, [0, 2], context_sizes=(1, 2, 3, 4))
    assert len(episodes) == 2 * (4 + 6 + 4)
    reduced = EpisodeDataset(ds, [0, 2], max_episodes_per_compound=3, seed=4)
    assert len(reduced) == 6
    batch = make_training_batch(ds, [0, 2], (0,), (1, 2, 3))
    assert batch["target_y"].shape == (2, 3, 4)
    assert "target_y" not in batch["inputs"]
    assert batch["inputs"]["chem"].shape == (2, 17)
    with pytest.raises(ValueError, match="already"):
        make_inference_batch(ds, [0], (0,), (0, 1))


def test_morgan_fingerprint_marks_missing_and_is_deterministic():
    a, mask, info = morgan_fingerprints(["CCO", "OCC", "", "not_a_smiles"], n_bits=64)
    assert a.shape == (4, 65)
    np.testing.assert_array_equal(a[0], a[1])
    assert mask.tolist() == [True, True, False, False]
    assert a[:, -1].tolist() == [1, 1, 0, 0]
    assert info["missing"] == 2


def test_schema_rejects_false_observation_and_invalid_groups():
    ds = records()
    bad = ds.Y.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replace(ds, Y=bad)
    with pytest.raises(ValueError, match="group"):
        replace(ds, feature_group_index=np.array([0, 0, 999, 1]))
