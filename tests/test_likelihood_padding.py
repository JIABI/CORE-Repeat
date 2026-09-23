"""Exact elimination of environmental factors unseen by the likelihood."""
import pytest
import torch

from opal2.model import JointGaussian
from test_model import dense_full_covariance


def padded_gaussian(*, zero_observed_loadings=False, whole_compound_missing=False):
    torch.manual_seed(516)
    dtype = torch.float64
    b, t, d, rank = 2, 4, 3, 2
    mean = torch.randn(b, t, d, dtype=dtype).requires_grad_()
    diagonal = (torch.rand(b, t, d, dtype=dtype) + .7).requires_grad_()
    local = (.3 * torch.randn(b, t, d, 3, dtype=dtype)).requires_grad_()
    groups = torch.tensor([
        [[0, 0, 0], [0, 1, 1], [-1, -1, -1], [-1, -1, -1]],
        [[0, 0, 0], [-1, -1, -1], [2, 5, 7], [-1, -1, -1]],
    ], dtype=torch.int64)
    mask = torch.tensor([
        [[True, True, True], [False, True, False], [False]*3, [False]*3],
        [[True, True, True], [True, True, True], [False]*3, [False]*3],
    ])
    if whole_compound_missing:
        mask[1] = False
    loads = []
    for _ in range(3):
        value = .4 * torch.randn(b, t, d, rank, dtype=dtype)
        if zero_observed_loadings:
            value[mask] = 0
        loads.append(value.requires_grad_())
    target = torch.randn(b, t, d, dtype=dtype)
    target[~mask] = float("nan")
    dist = JointGaussian(mean, diagonal, torch.cat((local, *loads), -1),
                         local, tuple(loads), groups)
    return dist, target, mask, (mean, diagonal, local, *loads)


def global_column_counts(dist, mask):
    all_columns, kept_columns = 0, 0
    for load, indices, n_groups in dist._environment_specs():
        all_columns += n_groups * load.shape[-1]
        kept_columns += torch.unique(indices[mask.any(-1)]).numel() * load.shape[-1]
    return all_columns, kept_columns


@pytest.mark.parametrize("zero_observed_loadings", [False, True])
@pytest.mark.parametrize("whole_compound_missing", [False, True])
def test_padding_compression_matches_uncompressed_and_dense_values_and_gradients(
        monkeypatch, zero_observed_loadings, whole_compound_missing):
    torch.set_num_threads(1)
    dist, target, mask, inputs = padded_gaussian(
        zero_observed_loadings=zero_observed_loadings,
        whole_compound_missing=whole_compound_missing)
    total, kept = global_column_counts(dist, mask)
    assert 0 < kept < total
    shapes = []
    original_cholesky = torch.linalg.cholesky
    def recording_cholesky(matrix, *args, **kwargs):
        shapes.append(tuple(matrix.shape))
        return original_cholesky(matrix, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg, "cholesky", recording_cholesky)
        actual = dist.joint_log_prob(target, mask)
    # Even zero-valued observed loadings keep their model latent coordinates.
    # Unknown groups belonging to actual observations are likewise retained.
    assert shapes[-1] == (kept, kept)
    uncompressed = dist._joint_log_prob_batched(target, mask)
    observed = mask.flatten()
    covariance = dense_full_covariance(dist)[observed][:, observed]
    dense = torch.distributions.MultivariateNormal(dist.mean.flatten()[observed],
                                                   covariance_matrix=covariance)
    expected = dense.log_prob(target.flatten()[observed])
    torch.testing.assert_close(actual, uncompressed, atol=1e-10, rtol=1e-11)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-11)
    gradient_actual = torch.autograd.grad(actual, inputs, retain_graph=True)
    gradient_old = torch.autograd.grad(uncompressed, inputs, retain_graph=True)
    gradient_dense = torch.autograd.grad(expected, inputs)
    for current, old, oracle in zip(gradient_actual, gradient_old, gradient_dense):
        torch.testing.assert_close(current, old, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(current, oracle, atol=1e-10, rtol=1e-9)
    # All unobserved coordinates are integrated out, not assigned fake data.
    for gradient in gradient_actual:
        expanded = mask if gradient.ndim == 3 else mask.unsqueeze(-1).expand_as(gradient)
        assert torch.count_nonzero(gradient[~expanded]) == 0


def test_unknown_observed_groups_are_not_merged_or_pruned(monkeypatch):
    dist, target, mask, _ = padded_gaussian()
    dist.environment_groups[:] = -1
    total, kept = global_column_counts(dist, mask)
    assert total == 2 * 4 * 3 * 2
    assert kept == int(mask.any(-1).sum()) * 3 * 2
    actual = dist.joint_log_prob(target, mask)
    # Every unknown environment is private to its actual physical well.
    torch.testing.assert_close(actual, dist._joint_log_prob_batched(target, mask), atol=1e-10, rtol=1e-11)


def test_entirely_unobserved_target_returns_zero_without_cholesky(monkeypatch):
    dist, target, mask, _ = padded_gaussian()
    mask[:] = False
    target[:] = float("nan")
    def forbidden(*args, **kwargs):
        raise AssertionError("No observed coordinates require no factorization")
    monkeypatch.setattr(torch.linalg, "cholesky", forbidden)
    result = dist.joint_log_prob(target, mask)
    assert result.item() == 0
    result.backward()
    assert torch.count_nonzero(dist.mean.grad) == 0


def test_likelihood_call_does_not_change_future_covariance_or_sampling():
    dist, target, mask, _ = padded_gaussian()
    before = dist.sample_joint(9, torch.Generator().manual_seed(714))
    factors = tuple(value.detach().clone() for value in dist.environment_loadings)
    groups = dist.environment_groups.clone()
    dist.joint_log_prob(target, mask)
    after = dist.sample_joint(9, torch.Generator().manual_seed(714))
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert torch.equal(groups, dist.environment_groups)
    for original, current in zip(factors, dist.environment_loadings):
        torch.testing.assert_close(original, current, atol=0, rtol=0)
