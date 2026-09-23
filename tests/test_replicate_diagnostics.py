"""Synthetic checks of diagnostic arithmetic, not empirical study evidence."""
import json

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import norm
import torch

from opal2.model import JointGaussian
from opal2.replicate_diagnostics import (
    gaussian_contrast_moments,
    summarize_contrasts,
)


def _hierarchical_example():
    generator = torch.Generator().manual_seed(919)
    b, t, d = 2, 3, 4
    mean = torch.randn(b, t, d, generator=generator, dtype=torch.float64)
    diagonal = .2 + torch.rand(b, t, d, generator=generator, dtype=torch.float64)
    local = torch.randn(b, t, d, 2, generator=generator, dtype=torch.float64) * .3
    groups = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 0, 1], [1, 0, 0]]])
    loads = tuple(torch.randn(b, t, d, 2, generator=generator, dtype=torch.float64) * a
                  for a in (.8, .5, .2))
    parts = [local]
    for level, load in enumerate(loads):
        expanded = torch.zeros(b, t, d, 2 * t, dtype=torch.float64)
        for i in range(b):
            assignments = {}
            for target in range(t):
                key = tuple(groups[i, target, :level + 1].tolist())
                group = assignments.setdefault(key, len(assignments))
                expanded[i, target, :, 2 * group:2 * group + 2] = load[i, target]
        parts.append(expanded)
    return JointGaussian(mean, diagonal, torch.cat(parts, dim=-1),
                         local, loads, groups)


def _dense_compound_covariance(distribution, object_index):
    t, d = distribution.mean.shape[1:]
    factor = distribution.local_factors[object_index].reshape(t * d, -1)
    covariance = torch.diag(distribution.diag_var[object_index].flatten()) + factor @ factor.T
    for level, loads in enumerate(distribution.environment_loadings):
        for a in range(t):
            for b in range(t):
                ga = distribution.environment_groups[object_index, a, :level + 1]
                gb = distribution.environment_groups[object_index, b, :level + 1]
                if a == b or ((ga >= 0).all() and torch.equal(ga, gb)):
                    covariance[a*d:(a+1)*d, b*d:(b+1)*d] += (
                        loads[object_index, a] @ loads[object_index, b].T)
    return covariance.numpy()


def test_exact_contrasts_match_dense_hierarchical_covariance_and_affine_means():
    distribution = _hierarchical_example()
    weights = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, -1, 0], [1, 0, -1], [0, 1, -1],
                        [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
                        [1/3, 1/3, 1/3], [2, -.5, 1.25]])
    center = np.array([10., -5., .3, 90.])
    scale = np.array([.3, 2., 1.3, 3.])
    means, variances = gaussian_contrast_moments(distribution, weights, center, scale)
    physical = distribution.mean.numpy() * scale + center
    expected_mean = np.einsum("ct,ntd->ncd", weights, physical)
    np.testing.assert_allclose(means, expected_mean, atol=1e-12, rtol=1e-12)
    for i in range(2):
        covariance = _dense_compound_covariance(distribution, i)
        for c, weight in enumerate(weights):
            transform = np.kron(weight[None, :], np.diag(scale))
            expected_variance = np.diag(transform @ covariance @ transform.T)
            np.testing.assert_allclose(variances[i, c], expected_variance, atol=1e-12, rtol=1e-12)


def test_common_factor_inflation_leaves_difference_variance_changes_average_variance():
    mean = torch.zeros(1, 3, 2, dtype=torch.float64)
    diagonal = torch.tensor([[[.2, .5], [.2, .5], [.2, .5]]], dtype=torch.float64)
    load = torch.tensor([.4, 1.1], dtype=torch.float64).reshape(1, 1, 2, 1).expand(1, 3, 2, 1)
    base = JointGaussian(mean, diagonal, load)
    inflated = JointGaussian(mean, diagonal, load * 4)
    weights = np.array([[1., -1., 0.], [1/3, 1/3, 1/3], [1, 0, 0]])
    _, v0 = gaussian_contrast_moments(base, weights, np.zeros(2), np.ones(2))
    _, v1 = gaussian_contrast_moments(inflated, weights, np.zeros(2), np.ones(2))
    np.testing.assert_array_equal(v0[:, 0], v1[:, 0])
    np.testing.assert_allclose(v0[0, 0], [2 * .2, 2 * .5])
    np.testing.assert_allclose(v1[0, 1] - v0[0, 1], 15 * np.array([.4, 1.1]) ** 2)
    assert (v1[:, 1:] > v0[:, 1:]).all()


def test_normal_quantiles_recover_nominal_coverage_width_and_interval_score():
    n, d = 100, 100
    z = ndtri((np.arange(n * d) + .5) / (n * d)).reshape(n, 1, d)
    scales = np.array([1., 3.]).reshape(1, 2, 1)
    mean = np.broadcast_to(np.array([10., -2.]).reshape(1, 2, 1), (n, 2, d))
    actual = mean + scales * z
    variance = np.broadcast_to(scales ** 2, actual.shape)
    levels = (.5, .8, .9, .95)
    summary, per_object = summarize_contrasts(actual, mean, variance, levels)
    assert per_object["coverage"].shape == (n, 2, 4)
    assert per_object["residual_sse"].shape == (n, 2)
    for index, row in enumerate(summary["contrasts"]):
        np.testing.assert_allclose(row["coverage"], levels, atol=1e-12)
        quantile = ndtri((1 + np.array(levels)) / 2)
        np.testing.assert_allclose(row["width"], 2 * scales[0, index, 0] * quantile, atol=1e-12)
        expected_score = scales[0, index, 0] * 4 * norm.pdf(quantile) / (1 - np.array(levels))
        np.testing.assert_allclose(row["interval_score"], expected_score, atol=.001, rtol=.001)
        assert abs(row["standardized_residual_mean"]) < 1e-12
        assert abs(row["standardized_residual_second_moment"] - 1) < .001
        assert abs(row["residual_rms_to_predicted_rms_sd"] - 1) < .001
        np.testing.assert_allclose(np.array(row["lower_tail"]) + row["upper_tail"]
                                   + row["coverage"], 1, atol=1e-12)
    json.dumps(summary, allow_nan=False)


def test_interval_score_tails_bias_and_scale_ratio_not_coverage():
    actual = np.array([[[-3., 0., 4.]]])
    mean = np.zeros_like(actual)
    variance = np.ones_like(actual)
    level = 2 * norm.cdf(2.) - 1
    summary, per_object = summarize_contrasts(actual, mean, variance, (level,))
    row = summary["contrasts"][0]
    np.testing.assert_allclose(row["coverage"], [1/3])
    np.testing.assert_allclose(row["lower_tail"], [1/3])
    np.testing.assert_allclose(row["upper_tail"], [1/3])
    np.testing.assert_allclose(row["interval_score"], [4 + 2 / (1 - level)])
    assert row["residual_mean"] == pytest.approx(1/3)
    assert row["residual_rms_to_predicted_rms_sd"] == pytest.approx(np.sqrt(25/3))
    assert per_object["residual_sse"][0, 0] == 25
    assert per_object["predicted_variance_sum"][0, 0] == 3


def test_all_3617_coordinates_remain_in_summary():
    actual = np.zeros((2, 10, 3617))
    actual[1, 9, -1] = 500
    summary, per_object = summarize_contrasts(actual, np.zeros_like(actual), np.ones_like(actual))
    assert summary["n_coordinates_per_contrast"] == 3617
    assert per_object["residual_sse"][1, 9] == 250000
    assert summary["contrasts"][9]["coordinate_mse"] == pytest.approx(250000 / (2 * 3617))
    assert per_object["coverage"][1, 9, -1] == pytest.approx(3616/3617)


@pytest.mark.parametrize("bad_levels", [(), (0.,), (1.,), (.9, .9), (float("nan"),)])
def test_invalid_levels_rejected(bad_levels):
    with pytest.raises(ValueError, match="levels"):
        summarize_contrasts(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), np.ones((1, 1, 2)), bad_levels)


@pytest.mark.parametrize("bad_variance", [0., -1., float("nan"), float("inf")])
def test_invalid_variance_rejected(bad_variance):
    with pytest.raises(ValueError):
        summarize_contrasts(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)),
                            np.full((1, 1, 2), bad_variance))


def test_invalid_shapes_weights_scalers_and_distribution_rejected():
    d = _hierarchical_example()
    with pytest.raises(TypeError):
        gaussian_contrast_moments(object(), np.ones((1, 3)), np.zeros(4), np.ones(4))
    for weights in (np.ones(3), np.ones((2, 2)), np.zeros((1, 3)), np.array([[1, np.nan, 0]])):
        with pytest.raises(ValueError):
            gaussian_contrast_moments(d, weights, np.zeros(4), np.ones(4))
    with pytest.raises(ValueError):
        gaussian_contrast_moments(d, np.ones((1, 3)), np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError):
        gaussian_contrast_moments(d, np.ones((1, 3)), np.zeros(3), np.ones(4))
    with pytest.raises(ValueError):
        summarize_contrasts(np.zeros((1, 1, 2)), np.zeros((1, 2, 2)), np.ones((1, 1, 2)))
    with pytest.raises(ValueError):
        summarize_contrasts(np.empty((0, 1, 2)), np.empty((0, 1, 2)), np.empty((0, 1, 2)))
