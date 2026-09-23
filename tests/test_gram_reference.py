"""Small synthetic adapter tests only; no real data or checkpoints are opened."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opal2.closed_form_baseline import ConditionalGaussian, fit_baseline
from opal2.data import MeasurementDataset, make_inference_batch
from opal2.gram_geometry import profiles_to_gram
from opal2.gram_reference import (ClosedFormGramReference, FrozenFaithfulGramReference,
                                 _snapshot_package, sample_neural_grams)
from opal2.model import EnvironmentNoiseCache, JointGaussian


def baseline_fixture():
    rng = np.random.default_rng(6)
    signal = rng.normal(size=(32, 1, 8))
    y = signal + rng.normal(size=(32, 4, 8)) + np.arange(4)[None, :, None] / 5
    return fit_baseline(y, k=4, clip=8.), y[:5, 0]


def test_closed_form_checkpoint_matches_joint_draws_and_mean_profiles(tmp_path):
    model, x = baseline_fixture()
    path = tmp_path / "closed_form.npz"
    model.save(path)
    reference = ClosedFormGramReference.from_checkpoint(path)
    result = reference.sample(x, 11, seed=51, object_chunk_size=5, draw_chunk_size=11)
    prediction = model.conditional(x[:, None], [0], [1, 2, 3])
    draws = prediction.sample_joint(11, seed=51)
    profiles = np.concatenate((np.broadcast_to(x[None, :, None], (11, 5, 1, 8)), draws), axis=-2)
    expected = profiles_to_gram(torch.from_numpy(profiles)).numpy()
    np.testing.assert_array_equal(result.grams, expected)
    point = np.concatenate((x[:, None], prediction.mean), axis=1)
    np.testing.assert_array_equal(result.mean_profile_gram, profiles_to_gram(torch.from_numpy(point)).numpy())
    assert result.grams.dtype == np.float64 and result.grams.shape == (11, 5, 4, 4)
    assert result.metadata["checkpoint"] == str(path)
    assert result.metadata["future_outcomes_used_as_inputs"] is False
    # E(Y Y^T) includes measurement uncertainty, so it is not E(Y) E(Y)^T.
    assert not np.allclose(result.grams.mean(0), result.conditional_mean_gram)


def test_closed_form_both_chunk_axes_bounded_and_reproducible(monkeypatch):
    model, x = baseline_fixture()
    calls = []
    original = ConditionalGaussian.sample_joint

    def record(self, n_samples, *args, **kwargs):
        calls.append((n_samples, len(self.mean)))
        return original(self, n_samples, *args, **kwargs)

    monkeypatch.setattr(ConditionalGaussian, "sample_joint", record)
    reference = ClosedFormGramReference(model)
    left = reference.sample(x, 13, seed=13, object_chunk_size=2, draw_chunk_size=4)
    right = reference.sample(x, 13, seed=13, object_chunk_size=2, draw_chunk_size=4)
    np.testing.assert_array_equal(left.grams, right.grams)
    assert max(a for a, b in calls) <= 4 and max(b for a, b in calls) <= 2
    assert len(calls) == 2 * 3 * 4
    np.testing.assert_allclose(left.grams[..., 0, 0], 1., atol=0, rtol=0)
    with pytest.raises(ValueError, match="array"):
        reference.sample(np.repeat(x[:, None], 4, axis=1), 2, seed=1)
    with pytest.raises(ValueError, match="Zero-norm"):
        reference.sample(np.zeros_like(x), 2, seed=1)
    with pytest.raises(ValueError, match="seed"):
        reference.sample(x, 2, seed=-1)


def environment_only_distribution(indices):
    b, t, d = len(indices), 3, 8
    mean = torch.arange(1, t * d + 1, dtype=torch.float32).reshape(1, t, d).expand(b, t, d) / 8
    zero = torch.zeros_like(mean)
    local = torch.zeros(b, t, d, 1)
    load = torch.ones(b, t, d, 1)
    groups = torch.zeros(b, t, 3, dtype=torch.long)
    return JointGaussian(mean, zero, load, local, (load, load * .5, load * .25), groups)


def test_neural_sampling_keeps_environment_shared_across_object_chunks():
    x = np.ones((5, 8))
    kwargs = dict(n_samples=17, seed=3, object_chunk_size=2, draw_chunk_size=5)
    inverse = lambda a: a * 2 + .1
    left = sample_neural_grams(x, environment_only_distribution, inverse, EnvironmentNoiseCache, **kwargs)
    right = sample_neural_grams(x, environment_only_distribution, inverse, EnvironmentNoiseCache, **kwargs)
    np.testing.assert_array_equal(left.grams, right.grams)
    # With only the common environment noise, identical objects must receive
    # identical draws even when they are evaluated in different object chunks.
    for obj in range(1, 5):
        np.testing.assert_array_equal(left.grams[:, 0], left.grams[:, obj])
    assert left.metadata["environment_cache_reused_across_object_chunks"]
    assert left.metadata["native_prediction_and_inverse_affine_precision_preserved"]
    assert left.grams.dtype == np.float64


def test_neural_single_block_matches_original_sampling_precision():
    x = np.ones((2, 8))
    distribution = environment_only_distribution(np.arange(2))
    inverse = lambda a: a * 1.234567 + .345678
    result = sample_neural_grams(x, environment_only_distribution, inverse,
        EnvironmentNoiseCache, 9, seed=22, object_chunk_size=2, draw_chunk_size=9)
    draws = inverse(distribution.sample_joint(9, torch.Generator().manual_seed(22 + 501), EnvironmentNoiseCache()))
    profiles = torch.cat((torch.ones(9, 2, 1, 8, dtype=torch.float64), draws.double()), -2)
    np.testing.assert_array_equal(result.grams, profiles_to_gram(profiles).numpy())
    point = torch.cat((torch.ones(2, 1, 8, dtype=torch.float64), inverse(distribution.mean).double()), -2)
    np.testing.assert_array_equal(result.conditional_mean_gram, profiles_to_gram(point).numpy())


def test_frozen_adapter_builds_context_only_and_is_future_value_invariant():
    n, w, d = 3, 4, 8
    y = np.arange(1, n * w * d + 1).reshape(n, w, d) / 10
    ds = MeasurementDataset(y, np.array(["a", "b", "c"]), np.array([f"f{i}" for i in range(d)]),
        np.zeros(d, int), ["all"], np.zeros((n, w, 2)), np.zeros((n, w, 3, 2)),
        np.zeros((n, w, 3), bool), np.zeros((n, w, 3), int), np.zeros((n, 2)),
        metadata={"fixture": True})
    adapter = FrozenFaithfulGramReference()
    adapter.X, adapter.ids, adapter.normalized = y[:, 0].copy(), ds.ids, ds
    adapter.config = SimpleNamespace(reference_access="observed_only", device="cpu")
    adapter._make_inputs = make_inference_batch
    adapter._ablate_inputs = lambda batch, config: batch
    adapter._cache_factory = EnvironmentNoiseCache
    adapter.scaler = SimpleNamespace(inverse_y=lambda a: a)
    adapter.provenance = {"fixture": True}
    seen = []

    def predictor(inputs):
        seen.append(inputs["context_y"].clone())
        assert inputs["context_y"].shape[1] == 1
        assert not bool(inputs["target_n_cells_mask"].any())
        assert not bool(inputs["target_reference_mask"].any())
        mean = inputs["context_y"].expand(-1, 3, -1).clone()
        return JointGaussian(mean, torch.ones_like(mean) * .2, torch.zeros(*mean.shape, 1))

    adapter.model = predictor
    left = adapter.sample([0, 2], 7, seed=18, object_chunk_size=1, draw_chunk_size=3)
    changed = y.copy()
    changed[:, 1:] = changed[:, 1:] * -10000 + 12345
    adapter.normalized = replace(ds, Y=changed)
    right = adapter.sample([0, 2], 7, seed=18, object_chunk_size=1, draw_chunk_size=3)
    np.testing.assert_array_equal(left.grams, right.grams)
    assert len(seen) == 4
    with pytest.raises(ValueError, match="Unique"):
        adapter.sample([0, 0], 2, seed=1)
    with pytest.raises(ValueError, match="Unique"):
        adapter.sample([3], 2, seed=1)


def test_snapshot_import_is_isolated_and_invalid_cohort_fails_before_load(tmp_path):
    import opal2
    source = tmp_path / "source_snapshot/opal2"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("FROZEN_FIXTURE = True\n")
    alias = _snapshot_package(source.parent)
    import importlib
    frozen = importlib.import_module(alias)
    assert frozen.FROZEN_FIXTURE and frozen is not opal2
    assert _snapshot_package(source.parent) == alias
    (tmp_path / "run_manifest.json").write_text(json.dumps({"data_shape": [999, 5, 3617]}))
    with pytest.raises(ValueError, match="already-opened"):
        FrozenFaithfulGramReference.from_run(tmp_path)
