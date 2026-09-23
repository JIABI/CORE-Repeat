"""Synthetic arithmetic/provenance tests, not real-data model evidence."""
import copy

import numpy as np
import pytest
import torch

from opal2.closed_form_baseline import ClosedFormBaseline
from opal2.data import TrainScaler
from opal2.model import JointGaussian
from opal2.moment_repair import ClosedFormGaussianAdapter
from opal2.statistical_repair import (
    AffineConditionalGaussian,
    ResidualMeanCorrection,
    StatisticalRepairAdapter,
    fit_residual,
)


def _baseline_and_data(d=5, rank=3, n=30):
    rng = np.random.default_rng(381)
    ids = tuple(f"train-{i}" for i in range(n))
    basis = np.linalg.qr(rng.normal(size=(d, rank)))[0]
    baseline = ClosedFormBaseline(
        center=np.linspace(-1, 1, d), scale=np.linspace(.8, 1.7, d),
        slot_mean=rng.normal(size=(4, d)) * .1, basis=basis,
        signal_cov=np.eye(rank) * .7, within_cov=np.eye(rank) * .3,
        residual_var=np.full(d, .2), reliability_vectors=np.eye(rank),
        reliability_eigenvalues=np.ones(rank),
        metadata={"schema_version": 1, "train_ids": list(ids), "clip": 8.})
    x = rng.normal(size=(n, d)) * baseline.scale + baseline.center
    mean = baseline.conditional(x[:, None], [0], [1, 2, 3]).mean
    k = min(64, rank)
    scores = baseline.transform_features(x, [0]) @ basis[:, :k]
    coef = np.stack([(.2 + .1 * j) * scores + j * .3 for j in range(3)], axis=1)
    delta = (coef @ basis[:, :k].T) * baseline.scale
    y = np.concatenate((x[:, None], mean + delta), axis=1)
    scaler = TrainScaler(baseline.center.copy(), baseline.scale.copy(), np.zeros(1), np.ones(1),
                         np.zeros((3, 1)), np.ones((3, 1)), list(ids), [f"f{i}" for i in range(d)])
    return baseline, scaler, y, ids, delta


def _batch(y, baseline):
    return {"context_y": torch.from_numpy((y[:, :1] - baseline.center) / baseline.scale),
            "context_mask": torch.ones(len(y), 1, dtype=torch.bool),
            "target_cond": torch.zeros(len(y), 3, 1, dtype=torch.float64),
            "target_group": torch.zeros(len(y), 3, 3, dtype=torch.int64)}


def _joint_example():
    generator = torch.Generator().manual_seed(602)
    b, t, d = 2, 3, 2
    mean = torch.randn(b, t, d, generator=generator, dtype=torch.float64)
    diagonal = .5 + torch.rand(b, t, d, generator=generator, dtype=torch.float64)
    local = torch.randn(b, t, d, 2, generator=generator, dtype=torch.float64) * .2
    loads = tuple(torch.randn(b, t, d, 1, generator=generator, dtype=torch.float64) * .3
                  for _ in range(3))
    # All wells share each environment; each object has its own local draw.
    groups = torch.zeros(b, t, 3, dtype=torch.int64)
    return JointGaussian(mean, diagonal, torch.cat((local, *loads), -1), local, loads, groups)


def _dense_covariance(distribution):
    b, t, d = distribution.mean.shape
    out = torch.diag(distribution.diag_var.flatten())
    local = distribution.local_factors
    for i in range(b):
        factor = local[i].reshape(t * d, -1)
        out[i*t*d:(i+1)*t*d, i*t*d:(i+1)*t*d] += factor @ factor.T
    for load in distribution.environment_loadings:
        factor = load.reshape(b * t * d, -1)
        out += factor @ factor.T
    return out


def test_residual_recovers_linear_role_specific_mean_and_train_only_feature_scaler():
    baseline, _, y, ids, expected = _baseline_and_data()
    frozen = copy.deepcopy(baseline)
    correction = fit_residual(baseline, y, ids, alpha=0)
    np.testing.assert_allclose(correction.predict(y[:, 0]), expected, atol=1e-12)
    affine = (y[:, 0] - baseline.center) / baseline.scale
    features = np.column_stack(((affine - baseline.slot_mean[0]) @ baseline.basis,
                                np.log1p(np.square(affine).mean(1))))
    np.testing.assert_allclose(correction.feature_center, features.mean(0))
    np.testing.assert_allclose(correction.feature_scale, features.std(0))
    for name in ("center", "scale", "basis", "signal_cov", "within_cov", "residual_var", "slot_mean"):
        np.testing.assert_array_equal(getattr(baseline, name), getattr(frozen, name))
    assert correction.coefficients.shape == (baseline.rank + 1, 3, baseline.rank)


def test_ridge_matches_unpenalized_intercept_sum_objective():
    baseline, _, y, ids, _ = _baseline_and_data()
    alpha = 12.
    correction = fit_residual(baseline, y, ids, alpha=alpha)
    features = (correction._features(y[:, 0]) - correction.feature_center) / correction.feature_scale
    target = ((y[:, 1:] - baseline.conditional(y[:, :1], [0], [1, 2, 3]).mean)
              / baseline.scale) @ correction.basis
    centered = (target - target.mean(0)).reshape(len(y), -1)
    expected = np.linalg.solve(features.T @ features + alpha * np.eye(features.shape[1]),
                               features.T @ centered)
    np.testing.assert_allclose(correction.coefficients.reshape(features.shape[1], -1), expected)
    np.testing.assert_allclose(correction.intercept, target.mean(0))


def test_rank_cap_and_full_space_covariance_are_retained():
    baseline, scaler, y, ids, _ = _baseline_and_data(d=70, rank=66, n=80)
    correction = fit_residual(baseline, y, ids, alpha=100)
    assert correction.rank == 64
    assert correction.coefficients.shape == (65, 3, 64)
    delta = correction.predict(y[:2, 0])
    assert delta.shape == (2, 3, 70)
    standardized_delta = delta / baseline.scale
    projection = standardized_delta @ correction.basis @ correction.basis.T
    np.testing.assert_allclose(projection, standardized_delta, atol=1e-12)
    batch = _batch(y[:2], baseline)
    base = ClosedFormGaussianAdapter(baseline, scaler)(batch)
    repaired = StatisticalRepairAdapter(baseline, scaler, correction)(batch)
    assert repaired.mean.shape[-1] == 70
    torch.testing.assert_close(repaired.diag_var, base.diag_var, atol=0, rtol=0)
    torch.testing.assert_close(repaired.factors, base.factors, atol=0, rtol=0)
    assert bool((repaired.diag_var > 0).all())


def test_serialization_prediction_identity_and_no_future_input(tmp_path):
    baseline, scaler, y, ids, _ = _baseline_and_data()
    correction = fit_residual(baseline, y, ids, alpha=1)
    path = tmp_path / "residual.npz"
    correction.save(path)
    restored = ResidualMeanCorrection.load(path)
    np.testing.assert_array_equal(correction.predict(y[:, 0]), restored.predict(y[:, 0]))
    assert restored.train_ids == ids
    original_prediction = restored.predict(y[:, 0])
    mutated = y.copy()
    mutated[:, 1:] = 1e7
    np.testing.assert_array_equal(restored.predict(mutated[:, 0]), original_prediction)
    model = StatisticalRepairAdapter(baseline, scaler, restored)
    batch = _batch(y[:2], baseline)
    before = model(batch).mean.clone()
    batch["target_y"] = torch.from_numpy(mutated[:2, 1:])
    with pytest.raises(ValueError, match="Future"):
        model(batch)
    del batch["target_y"]
    torch.testing.assert_close(model(batch).mean, before, atol=0, rtol=0)
    with pytest.raises(ValueError):
        restored.predict(y)  # All four roles are not an inference input.


def test_neutral_and_weight_zero_exact_base_identity():
    baseline, scaler, y, ids, _ = _baseline_and_data()
    correction = fit_residual(baseline, y, ids, 1)
    batch = _batch(y[:2], baseline)
    base = ClosedFormGaussianAdapter(baseline, scaler)(batch)
    repaired = StatisticalRepairAdapter(baseline, scaler, correction, a=1, residual_weight=0)(batch)
    target = torch.from_numpy((y[:2, 1:] - baseline.center) / baseline.scale)
    torch.testing.assert_close(repaired.mean, base.mean, atol=0, rtol=0)
    torch.testing.assert_close(repaired.log_prob(target), base.log_prob(target), atol=0, rtol=0)
    torch.testing.assert_close(
        repaired.sample_joint(4, torch.Generator().manual_seed(1)),
        base.sample_joint(4, torch.Generator().manual_seed(1)), atol=0, rtol=0)
    neutral = AffineConditionalGaussian(base, a=1)
    assert neutral.mean is base.mean
    assert neutral.factors is base.factors
    torch.testing.assert_close(neutral.log_prob(target), base.log_prob(target), atol=0, rtol=0)
    assert not np.any(correction.predict(y[:, 0], weight=0))


@pytest.mark.parametrize("mask_kind", ["all", "coordinate", "well", "all_missing_one_object"])
def test_affine_transport_log_density_matches_dense_gaussian_with_masks(mask_kind):
    base = _joint_example()
    shift = torch.linspace(-.4, .6, base.mean.numel()).reshape_as(base.mean)
    a = 2.3
    repaired = AffineConditionalGaussian(base, shift, a)
    target = repaired.mean + .1
    mask = torch.ones_like(target, dtype=torch.bool)
    if mask_kind == "coordinate":
        mask[0, 1, 0] = False
        target[1, 2, 1] = float("nan")
    elif mask_kind == "well":
        mask = torch.tensor([[True, False, True], [False, True, True]])
    elif mask_kind == "all_missing_one_object":
        mask[0] = False
    observed = repaired.observed_mask(target, mask)
    covariance = _dense_covariance(base) * a
    actual = repaired.log_prob(target, mask)
    for i in range(2):
        selected = observed[i].flatten()
        if not selected.any():
            assert actual[i] == 0
            continue
        block = covariance[i*6:(i+1)*6, i*6:(i+1)*6][selected][:, selected]
        expected = torch.distributions.MultivariateNormal(
            repaired.mean[i].flatten()[selected], covariance_matrix=block).log_prob(
                target[i].flatten()[selected])
        torch.testing.assert_close(actual[i], expected, atol=1e-11, rtol=1e-11)
    selected = observed.flatten()
    expected_joint = torch.distributions.MultivariateNormal(
        repaired.mean.flatten()[selected], covariance_matrix=covariance[selected][:, selected]
    ).log_prob(target.flatten()[selected])
    torch.testing.assert_close(repaired.joint_log_prob(target, mask), expected_joint, atol=1e-11, rtol=1e-11)


def test_affine_sample_covariance_and_signed_weighted_moments():
    base = _joint_example()
    shift = torch.ones_like(base.mean) * .25
    a = .7
    repaired = AffineConditionalGaussian(base, shift, a)
    covariance = _dense_covariance(base) * a
    torch.testing.assert_close(_dense_covariance(repaired), covariance, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(repaired.marginal_variance.flatten(), covariance.diagonal())
    weights = torch.tensor([[1., -1., 0.], [.5, 0., .5]], dtype=torch.float64)
    mean, diag, factor = repaired.weighted_moments(weights)
    for i in range(2):
        transform = torch.kron(weights[i:i+1], torch.eye(2, dtype=torch.float64))
        expected_cov = transform @ covariance[i*6:(i+1)*6, i*6:(i+1)*6] @ transform.T
        torch.testing.assert_close(mean[i], transform @ repaired.mean[i].flatten())
        torch.testing.assert_close(torch.diag(diag[i]) + factor[i] @ factor[i].T,
                                   expected_cov, atol=1e-12, rtol=1e-12)
    samples = repaired.sample_joint(20000, torch.Generator().manual_seed(43)).reshape(20000, -1)
    mean_se = torch.sqrt(covariance.diagonal() / len(samples))
    assert ((samples.mean(0) - repaired.mean.flatten()).abs() / mean_se).max() < 6
    covariance_se = torch.sqrt((covariance.diagonal()[:, None] * covariance.diagonal()[None, :]
                                + covariance.square()) / (len(samples) - 1))
    assert ((torch.cov(samples.T) - covariance).abs() / covariance_se).max() < 6


def test_adapter_mean_only_changes_no_covariance_and_scale_only_changes_no_mean():
    baseline, scaler, y, ids, _ = _baseline_and_data()
    correction = fit_residual(baseline, y, ids, 1)
    batch = _batch(y[:2], baseline)
    base = ClosedFormGaussianAdapter(baseline, scaler)(batch)
    mean_only = StatisticalRepairAdapter(baseline, scaler, correction)(batch)
    expected_shift = torch.from_numpy(correction.predict(y[:2, 0]) / baseline.scale)
    torch.testing.assert_close(mean_only.mean, base.mean + expected_shift)
    torch.testing.assert_close(mean_only.factors, base.factors, atol=0, rtol=0)
    torch.testing.assert_close(mean_only.diag_var, base.diag_var, atol=0, rtol=0)
    scaled = StatisticalRepairAdapter(baseline, scaler, a=3)(batch)
    torch.testing.assert_close(scaled.mean, base.mean, atol=0, rtol=0)
    torch.testing.assert_close(scaled.marginal_variance, base.marginal_variance * 3)


def test_invalid_scalars_fit_ids_and_coordinates_are_rejected():
    baseline, scaler, y, ids, _ = _baseline_and_data()
    for a in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            StatisticalRepairAdapter(baseline, scaler, a=a)
    for alpha in (-1, float("nan")):
        with pytest.raises(ValueError):
            fit_residual(baseline, y, ids, alpha)
    with pytest.raises(ValueError, match="outside"):
        fit_residual(baseline, y, ("evaluation-id", *ids[1:]), 1)
    correction = fit_residual(baseline, y, ids, 1)
    with pytest.raises(ValueError):
        correction.predict(y[:, 0], weight=1.1)
    correction.center[0] += 1
    with pytest.raises(ValueError, match="coordinates"):
        StatisticalRepairAdapter(baseline, scaler, correction)
