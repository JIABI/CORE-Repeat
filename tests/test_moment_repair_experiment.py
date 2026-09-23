"""Geometry reporting checks on synthetic arrays, not empirical evidence."""
import gc
import json
import weakref

import numpy as np
import torch

from opal2.moment_repair_experiment import GeometryModel


class _Scaler:
    def inverse_y(self, value):
        return value * value.new_tensor([2., 3.]) + value.new_tensor([.5, -1.])


class _Distribution:
    def __init__(self, mean, samples):
        self.mean = mean
        self.samples = samples
        self.position = 0

    def sample_joint(self, n_samples, generator=None, environment_noise_cache=None):
        start = self.position
        self.position += n_samples
        return self.samples[start:self.position]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.references = []

    def forward(self, batch):
        distribution = _Distribution(batch["mean"], batch["samples"])
        self.references.append(weakref.ref(distribution))
        return distribution


def _pair_cosines(values):
    unit = values / np.linalg.norm(values, axis=-1, keepdims=True)
    return np.stack([(unit[..., a, :] * unit[..., b, :]).sum(-1)
                     for a, b in ((0, 1), (0, 2), (1, 2))], axis=-1)


def test_geometry_uses_original_units_and_pools_objects_and_samples(tmp_path):
    rng = np.random.default_rng(932)
    samples = torch.from_numpy(rng.normal(size=(5, 3, 3, 2)) + [2., 1.])
    mean = torch.from_numpy(rng.normal(size=(3, 3, 2)) + [1., 2.])
    scaler = _Scaler()
    wrapper = GeometryModel(_Model(), scaler)
    # Unequal object chunks catch accidental averaging of chunk-level means.
    for start, stop in ((0, 2), (2, 3)):
        distribution = wrapper({"mean": mean[start:stop],
                                "samples": samples[:, start:stop]})
        torch.testing.assert_close(distribution.mean, mean[start:stop])
        torch.testing.assert_close(distribution.sample_joint(2), samples[:2, start:stop])
        torch.testing.assert_close(distribution.sample_joint(3), samples[2:, start:stop])
    original_samples = scaler.inverse_y(samples).numpy()
    original_mean = scaler.inverse_y(mean).numpy()
    actual = np.concatenate((np.ones((3, 1, 2)), original_samples[0]), axis=1)
    wrapper.save_geometry(tmp_path, actual)
    report = json.loads((tmp_path / "geometry.json").read_text())
    sample_cosines = _pair_cosines(original_samples)
    expected_mean_cosines = _pair_cosines(original_mean).mean(0)
    expected_sample_cosines = sample_cosines.reshape(-1, 3).mean(0)
    assert not np.allclose(expected_mean_cosines, expected_sample_cosines)
    np.testing.assert_allclose(report["mean_vector_cosine"], expected_mean_cosines)
    np.testing.assert_allclose(report["full_sample_cosine_mean"], expected_sample_cosines)
    np.testing.assert_allclose(report["full_sample_cosine_sd"],
                               sample_cosines.reshape(-1, 3).std(0))
    np.testing.assert_allclose(report["realised_well_cosine_mean"],
                               _pair_cosines(actual[:, 1:]).mean(0))
    np.testing.assert_allclose(report["full_sample_rms_mean"],
                               np.sqrt(np.square(original_samples).mean(-1)).mean((0, 1)))
    np.testing.assert_allclose(report["realised_rms_mean"],
                               np.sqrt(np.square(actual[:, 1:]).mean(-1)).mean(0))
    assert not wrapper.records


def test_geometry_records_do_not_retain_full_distributions():
    model = _Model()
    wrapper = GeometryModel(model, _Scaler())
    for _ in range(3):
        distribution = wrapper({"mean": torch.ones(2, 3, 2, dtype=torch.float64),
                                "samples": torch.ones(2, 2, 3, 2, dtype=torch.float64)})
        distribution.sample_joint(2)
        reference = model.references[-1]
        assert reference() is not None
        del distribution
        gc.collect()
        assert reference() is None
    assert len(wrapper.records) == 3
    assert all(reference() is None for reference in model.references)
