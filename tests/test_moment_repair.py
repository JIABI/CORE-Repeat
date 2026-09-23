"""Small synthetic engineering checks; these are not empirical model evidence."""
import copy

import numpy as np
import pytest
import torch

from opal2.closed_form_baseline import ClosedFormBaseline
from opal2.data import TrainScaler
from opal2.model import JointGaussian
from opal2.moment_repair import (
    ClosedFormGaussianAdapter,
    MomentRepairModel,
    conditional_to_joint_gaussian,
    repair_joint_gaussian,
)


def _dense_covariance(distribution):
    """Expand local and nested environment covariances independently of Woodbury."""
    b, t, d = distribution.mean.shape
    covariance = torch.diag(distribution.diag_var.flatten())
    local = (distribution.factors if distribution.local_factors is None
             else distribution.local_factors)
    for i in range(b):
        rows = slice(i * t * d, (i + 1) * t * d)
        factor = local[i].reshape(t * d, -1)
        covariance[rows, rows] += factor @ factor.T
    if distribution.environment_loadings is not None:
        for level, load in enumerate(distribution.environment_loadings):
            for i in range(b * t):
                bi, ti = divmod(i, t)
                left = distribution.environment_groups[bi, ti, :level + 1]
                for j in range(b * t):
                    bj, tj = divmod(j, t)
                    right = distribution.environment_groups[bj, tj, :level + 1]
                    if i == j or ((left >= 0).all() and torch.equal(left, right)):
                        covariance[i*d:(i+1)*d, j*d:(j+1)*d] += (
                            load[bi, ti] @ load[bj, tj].T)
    return covariance


def _original_distribution():
    generator = torch.Generator().manual_seed(192)
    b, t, d, latent_rank = 2, 3, 3, 2
    mean = torch.randn(b, t, d, generator=generator, dtype=torch.float64)
    diagonal = torch.full_like(mean, .45)
    local = torch.zeros(b, t, d, latent_rank + t, dtype=torch.float64)
    local[..., :latent_rank] = .35 * torch.randn(
        b, t, d, latent_rank, generator=generator, dtype=torch.float64)
    for target in range(t):
        local[:, target, :, latent_rank + target] = .25 * torch.randn(
            b, d, generator=generator, dtype=torch.float64)
    groups = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 0, 2], [1, 0, 0]]])
    loads = tuple(amplitude * torch.randn(b, t, d, 1, generator=generator,
                                          dtype=torch.float64)
                  for amplitude in (.35, .25, .20))
    marginal_parts = [local]
    for level, load in enumerate(loads):
        expanded = torch.zeros(b, t, d, t, dtype=torch.float64)
        for i in range(b):
            assigned = {}
            for target in range(t):
                key = tuple(groups[i, target, :level + 1].tolist())
                column = assigned.setdefault(key, len(assigned))
                expanded[i, target, :, column] = load[i, target, :, 0]
        marginal_parts.append(expanded)
    return JointGaussian(mean, diagonal, torch.cat(marginal_parts, -1),
                         local, loads, groups, environment_cache_namespace="fixture")


class _FixedModel(torch.nn.Module):
    latent_rank = 2

    def __init__(self, distribution):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64),
                                         requires_grad=False)
        self.distribution = distribution
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return self.distribution


@pytest.fixture
def example():
    baseline = ClosedFormBaseline(
        center=np.array([2., -1., 3.]), scale=np.array([.7, 1.3, 2.1]),
        slot_mean=np.array([[.2, -.1, .4], [.1, .3, -.2],
                            [-.2, .1, .1], [.4, -.3, .2]]),
        basis=np.eye(3)[:, :2], signal_cov=np.array([[.8, .15], [.15, .5]]),
        within_cov=np.array([[.35, .06], [.06, .25]]),
        residual_var=np.array([.12, .2, .3]),
        reliability_vectors=np.eye(2), reliability_eigenvalues=np.array([2., 1.]),
        metadata={"schema_version": 1, "train_ids": ["train-a", "train-b"]},
    )
    scaler = TrainScaler(
        baseline.center.copy(), baseline.scale.copy(), np.zeros(1), np.ones(1),
        np.zeros((3, 1)), np.ones((3, 1)), ["train-a", "train-b"], ["f0", "f1", "f2"])
    normalized = torch.tensor([[[1.2, -.8, .7]], [[-.4, .1, 1.6]]], dtype=torch.float64)
    physical = scaler.inverse_y(normalized.numpy())
    conditional = baseline.conditional(physical, [0], [1, 2, 3])
    original = _original_distribution()
    batch = {"context_y": normalized, "context_mask": torch.ones(2, 1, dtype=torch.bool),
             "target_cond": torch.zeros(2, 3, 1, dtype=torch.float64),
             "target_group": original.environment_groups.clone()}
    return baseline, scaler, conditional, original, batch


def test_conversion_matches_physical_and_affine_density_and_covariance(example):
    baseline, scaler, conditional, _, _ = example
    target = conditional.mean + np.linspace(-.3, .4, conditional.mean.size).reshape(conditional.mean.shape)
    scale = np.tile(baseline.scale, 3)
    for affine in (False, True):
        distribution = conditional_to_joint_gaussian(conditional, affine=affine)
        expected_mean = scaler.transform_y(conditional.mean) if affine else conditional.mean
        expected_covariance = conditional.dense_covariance()
        expected_log_prob = conditional.log_prob(target)
        observed = target
        if affine:
            expected_covariance = expected_covariance / scale[:, None] / scale[None, :]
            expected_log_prob = expected_log_prob + 3 * np.log(baseline.scale).sum()
            observed = scaler.transform_y(target)
        np.testing.assert_allclose(distribution.mean.numpy(), expected_mean, atol=1e-12)
        expected_full = torch.block_diag(*[torch.from_numpy(expected_covariance)] * 2)
        torch.testing.assert_close(_dense_covariance(distribution), expected_full,
                                   atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(distribution.log_prob(torch.from_numpy(observed)).numpy(),
                                   expected_log_prob, atol=1e-10, rtol=1e-10)
        assert distribution.mean.dtype == torch.float64


def test_mean_only_preserves_every_covariance_component(example):
    _, scaler, conditional, original, _ = example
    old_mean = original.mean.clone()
    repaired = repair_joint_gaussian(original, conditional, latent_rank=2, mode="mean_only")
    torch.testing.assert_close(repaired.mean, torch.from_numpy(scaler.transform_y(conditional.mean)))
    for name in ("diag_var", "factors", "local_factors", "environment_groups"):
        assert torch.equal(getattr(repaired, name), getattr(original, name))
    for actual, expected in zip(repaired.environment_loadings, original.environment_loadings):
        assert torch.equal(actual, expected)
    assert repaired.environment_cache_namespace == original.environment_cache_namespace
    torch.testing.assert_close(_dense_covariance(repaired), _dense_covariance(original), atol=0, rtol=0)
    assert torch.equal(original.mean, old_mean)


def test_within_repair_replaces_only_independent_noise(example):
    baseline, scaler, conditional, original, _ = example
    repaired = repair_joint_gaussian(original, conditional, latent_rank=2, mode="mean_and_within")
    expected = _dense_covariance(original).clone()
    noise = torch.from_numpy(np.diag(baseline.residual_var)
                             + baseline._noise_factor @ baseline._noise_factor.T)
    for i in range(2):
        rows = slice(i * 9, (i + 1) * 9)
        residual = original.local_factors[i, ..., 2:].reshape(9, -1)
        expected[rows, rows] -= torch.diag(original.diag_var[i].flatten()) + residual @ residual.T
        expected[rows, rows] += torch.block_diag(noise, noise, noise)
    torch.testing.assert_close(_dense_covariance(repaired), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(repaired.mean, torch.from_numpy(scaler.transform_y(conditional.mean)))
    assert torch.equal(repaired.local_factors[..., :2], original.local_factors[..., :2])
    assert torch.equal(repaired.environment_groups, original.environment_groups)
    assert repaired.environment_cache_namespace == original.environment_cache_namespace
    for actual, prior in zip(repaired.environment_loadings, original.environment_loadings):
        assert torch.equal(actual, prior)
    for i in range(2):
        factor = repaired.factors[i].reshape(9, -1)
        marginal = torch.diag(repaired.diag_var[i].flatten()) + factor @ factor.T
        torch.testing.assert_close(marginal, expected[i*9:(i+1)*9, i*9:(i+1)*9],
                                   atol=1e-12, rtol=1e-12)
    target = repaired.mean + torch.linspace(-.7, .8, 18).reshape(2, 3, 3)
    expected_logp = torch.stack([
        torch.distributions.MultivariateNormal(
            repaired.mean[i].flatten(), covariance_matrix=expected[i*9:(i+1)*9, i*9:(i+1)*9]
        ).log_prob(target[i].flatten()) for i in range(2)])
    torch.testing.assert_close(repaired.log_prob(target), expected_logp, atol=1e-10, rtol=1e-10)
    full_logp = torch.distributions.MultivariateNormal(
        repaired.mean.flatten(), covariance_matrix=expected).log_prob(target.flatten())
    torch.testing.assert_close(repaired.joint_log_prob(target), full_logp, atol=1e-10, rtol=1e-10)


def test_repaired_sampling_retains_cross_object_environment_moments(example):
    _, _, conditional, original, _ = example
    repaired = repair_joint_gaussian(original, conditional, latent_rank=2, mode="mean_and_within")
    expected = _dense_covariance(repaired)
    assert expected[:9, 9:].abs().max() > .01
    n = 20000
    samples = repaired.sample_joint(n, generator=torch.Generator().manual_seed(318)).reshape(n, -1)
    mean_se = (expected.diagonal() / n).sqrt()
    assert ((samples.mean(0) - repaired.mean.flatten()).abs() / mean_se).max() < 6
    covariance_se = ((expected.diagonal()[:, None] * expected.diagonal()[None, :]
                      + expected.square()) / (n - 1)).sqrt()
    assert ((torch.cov(samples.T) - expected).abs() / covariance_se).max() < 6


def test_adapters_use_fixed_context_and_return_normalized_coordinates(example):
    baseline, scaler, conditional, original, batch = example
    adapter = ClosedFormGaussianAdapter(baseline, scaler)
    distribution = adapter(batch)
    reference = conditional_to_joint_gaussian(conditional)
    torch.testing.assert_close(distribution.mean, reference.mean)
    torch.testing.assert_close(_dense_covariance(distribution), _dense_covariance(reference))
    base_model = _FixedModel(original)
    wrapper = MomentRepairModel(base_model, baseline, scaler, mode="mean_only", latent_rank=2)
    repaired = wrapper(batch)
    assert base_model.calls == 1
    torch.testing.assert_close(repaired.mean, reference.mean)
    torch.testing.assert_close(_dense_covariance(repaired), _dense_covariance(original))


def test_adapter_missing_target_likelihood_equals_dense_observed_marginal(example):
    baseline, scaler, _, _, batch = example
    distribution = ClosedFormGaussianAdapter(baseline, scaler)(batch)
    target = distribution.mean + .2
    target[0, 1, 2] = float("nan")
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[1, 2] = False
    observed = torch.isfinite(target) & mask
    full_covariance = _dense_covariance(distribution)
    expected = []
    for i in range(2):
        keep = observed[i].flatten()
        covariance = full_covariance[i*9:(i+1)*9, i*9:(i+1)*9][keep][:, keep]
        dense = torch.distributions.MultivariateNormal(distribution.mean[i].flatten()[keep],
                                                        covariance_matrix=covariance)
        expected.append(dense.log_prob(target[i].flatten()[keep]))
    expected = torch.stack(expected)
    torch.testing.assert_close(distribution.log_prob(target, mask), expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(distribution.joint_log_prob(target, mask), expected.sum(),
                               atol=1e-10, rtol=1e-10)


def test_both_wrappers_reject_future_target_measurements_before_base_call(example):
    baseline, scaler, _, original, batch = example
    base_model = _FixedModel(original)
    wrappers = [ClosedFormGaussianAdapter(baseline, scaler),
                MomentRepairModel(base_model, baseline, scaler, mode="mean_only", latent_rank=2)]
    for wrapper in wrappers:
        with pytest.raises(ValueError):
            wrapper(dict(batch, target_y=torch.zeros_like(original.mean)))
    assert base_model.calls == 0


def test_scaler_coordinate_or_training_identity_mismatch_is_rejected(example):
    baseline, scaler, _, original, _ = example
    for mismatch in ("y_center", "y_scale", "train_ids"):
        changed = copy.deepcopy(scaler)
        if mismatch == "train_ids":
            changed.train_ids = ["train-a", "different-training-object"]
        else:
            getattr(changed, mismatch)[0] = np.nextafter(getattr(changed, mismatch)[0], np.inf)
        with pytest.raises(ValueError):
            ClosedFormGaussianAdapter(baseline, changed)
        with pytest.raises(ValueError):
            MomentRepairModel(_FixedModel(original), baseline, changed, mode="mean_only", latent_rank=2)


def test_adapter_requires_one_available_context_and_three_future_wells(example):
    baseline, scaler, _, _, batch = example
    adapter = ClosedFormGaussianAdapter(baseline, scaler)
    invalid = [dict(batch, context_y=batch["context_y"].expand(-1, 2, -1),
                    context_mask=torch.ones(2, 2, dtype=torch.bool)),
               dict(batch, context_mask=torch.tensor([[True], [False]])),
               dict(batch, target_cond=torch.zeros(2, 2, 1, dtype=torch.float64))]
    for candidate in invalid:
        with pytest.raises(ValueError):
            adapter(candidate)
