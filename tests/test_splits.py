import numpy as np
import pytest
from opal2.splits import development_split, assert_disjoint_splits, save_split, load_split, source_split


def test_split_roundtrip(tmp_path):
    ids = np.array([f"c{i}" for i in range(639)])
    splits = development_split(ids)
    save_split(tmp_path / "split.json", ids, splits, evidence_scope="DEVELOPMENT_ONLY")
    loaded, scope = load_split(tmp_path / "split.json", ids)
    assert scope == "DEVELOPMENT_ONLY"
    assert all(np.array_equal(splits[k], loaded[k]) for k in splits)


def test_overlapping_split_rejected():
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_splits(["a", "b", "c"], {"train": [0, 1], "test": [1, 2]})


def test_source_split_is_before_pairs():
    s = np.array([["a", "b", "c", "d"]] * 3)
    parts = source_split(s, np.ones_like(s, bool), train_sources=["a"],
                         validation_sources=["b"], calibration_sources=["c"], evaluation_sources=["d"])
    assert parts["train"].sum() == 3
    assert not np.any(parts["train"] & parts["evaluation"])
