"""Synthetic isolation/serialization tests, not assay efficacy experiments."""
import numpy as np
import pytest
import torch

from opal2 import eu_r3_representation as module


def fixture():
    rng = np.random.default_rng(802)
    latent = rng.normal(size=(36, 12))
    profiles = latent[:, None]+rng.normal(size=(36, 4, 12))*.25
    profiles[24:, 1:] = np.nan  # Held-out future measurements are never needed.
    return dict(ids=np.asarray([f"id{i:03}" for i in range(36)]),
        groups=np.asarray([f"g{i//2:03}" for i in range(36)]), Y=profiles)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    path = tmp_path_factory.mktemp("eu_rep")/"fit"
    data = fixture()
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        out = module.fit_eu_states(data, np.arange(24), path, seed=17)
    finally:
        torch.set_num_threads(previous)
    return data, path, out


def test_full_configuration_train_isolation_and_no_future_input(fitted):
    data, _, out = fitted
    assert out["report"]["full_neural_configuration"] == module.NEURAL_CONFIG
    assert out["report"]["train_ids"] == data["ids"][:24].tolist()
    for name in module.ARMS:
        z, scale = out["states"][name]
        assert z.shape == (36, 8) and scale.shape == (8,)
        assert np.isfinite(z).all() and np.all(scale > 0)
        np.testing.assert_allclose(z[:24].mean(0), 0., atol=1e-14)
        assert np.ptp(scale) == 0
    for name in ("DIRECT_STATE", "CONDITIONAL_STATE"):
        r = out["report"]["arms"][name]["report"]
        assert r["epochs_requested"] == 60
        assert r["dimensions"]["latent"] == 8
        assert set(r["inner_fit_ids"]) | set(r["inner_validation_ids"]) == set(data["ids"][:24])
        assert not set(r["inner_fit_groups"]) & set(r["inner_validation_groups"])
    assert out["report"]["raw_images_used"] is False


def test_save_load_first_well_only(fitted):
    data, path, fit = fitted
    loaded = module.load_eu_states(path, data["Y"][:, 0])
    for name in module.ARMS:
        # PCA's fit_transform/transform centering order can differ by roundoff.
        np.testing.assert_allclose(loaded["states"][name][0], fit["states"][name][0], atol=1e-14, rtol=1e-14)
        np.testing.assert_array_equal(loaded["states"][name][1], fit["states"][name][1])
    with pytest.raises(ValueError, match="X must"):
        module.load_eu_states(path, data["Y"])


def test_group_overlap_rejected_before_creation(tmp_path):
    data = fixture()
    data["groups"][24] = data["groups"][0]
    path = tmp_path/"bad"
    with pytest.raises(ValueError, match="group crosses"):
        module.fit_eu_states(data, np.arange(24), path, seed=17)
    assert not path.exists()


def test_off_feature_path_is_exact_without_touching_state():
    base = np.arange(15, dtype=np.float32).reshape(5, 3)
    result = module.optional_state_features(base, np.full((5, 8), np.nan), enabled=False)
    np.testing.assert_array_equal(result, base)
    assert result.dtype == base.dtype and not np.shares_memory(result, base)
    joined = module.optional_state_features(base, np.ones((5, 8)), enabled=True)
    np.testing.assert_array_equal(joined[:, :3], base)
    assert joined.shape == (5, 11)
