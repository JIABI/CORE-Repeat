import copy
import pytest
import torch

from opal2.model import EnvironmentNoiseCache, GroupedProfileEncoder, JointGaussian, MeasurementWorldModel


def make_batch(b=3, c=2, t=3, d=7):
    torch.manual_seed(8)
    return {
        "context_y": torch.randn(b, c, d), "context_cond": torch.randn(b, c, 4),
        "context_mask": torch.ones(b, c, dtype=torch.bool),
        "context_reference": torch.randn(b, c, 3, 5),
        "context_reference_mask": torch.ones(b, c, 3, dtype=torch.bool),
        "context_group": torch.zeros(b, c, 3, dtype=torch.int64),
        "target_cond": torch.randn(b, t, 4), "target_reference": torch.randn(b, t, 3, 5),
        "target_reference_mask": torch.ones(b, t, 3, dtype=torch.bool),
        "target_group": torch.tensor([[[0, 0, 0], [0, 0, 1], [1, 0, 0]]]).expand(b, t, 3).clone(),
        "chem": torch.randn(b, 6),
    }


def new_model(**kw):
    return MeasurementWorldModel({"Nuclei_Intensity": [0, 1, 2], "Cells_Texture": [3, 4, 5, 6]},
                                 4, 5, 6, hidden_dim=16, latent_rank=2, residual_rank=1, **kw)


def dense_covariance(distribution, batch=0):
    f = distribution.factors[batch].flatten(0, 1)
    return torch.diag(distribution.diag_var[batch].flatten()) + f @ f.T


def dense_full_covariance(distribution):
    specs = distribution._environment_specs()
    if not specs:
        return torch.block_diag(*[dense_covariance(distribution, i)
                                  for i in range(distribution.mean.shape[0])])
    blocks = []
    global_blocks = []
    for i in range(distribution.mean.shape[0]):
        local = distribution.local_factors[i].flatten(0, 1)
        blocks.append(torch.diag(distribution.diag_var[i].flatten()) + local @ local.T)
        global_blocks.append(distribution._global_factor_block(i, specs))
    global_factor = torch.cat(global_blocks, 0)
    return torch.block_diag(*blocks) + global_factor @ global_factor.T


def test_joint_gaussian_matches_dense_likelihood_and_missing_marginal():
    torch.manual_seed(10)
    mean = torch.randn(2, 3, 4, dtype=torch.float64)
    diagonal = torch.rand_like(mean) + 0.3
    factors = torch.randn(2, 3, 4, 5, dtype=torch.float64) * 0.2
    joint = JointGaussian(mean, diagonal, factors)
    y = torch.randn_like(mean)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, 1, 2] = False
    mask[1, 2, :] = False
    y[~mask] = float("nan")
    actual = joint.log_prob(y, mask)
    for b in range(2):
        flat_mask = mask[b].flatten()
        cov = dense_covariance(joint, b)[flat_mask][:, flat_mask]
        dense = torch.distributions.MultivariateNormal(mean[b].flatten()[flat_mask], covariance_matrix=cov)
        assert torch.allclose(actual[b], dense.log_prob(y[b].flatten()[flat_mask]), atol=1e-10)


def test_exact_weighted_covariance_propagation():
    torch.manual_seed(11)
    joint = JointGaussian(torch.randn(2, 3, 4).double(), torch.rand(2, 3, 4).double() + 0.2,
                          torch.randn(2, 3, 4, 5).double())
    weights = torch.tensor([[0.2, 0.3, 0.5], [1., -1., 0.]], dtype=torch.float64)
    mean, diag, factor = joint.weighted_moments(weights)
    for b in range(2):
        transform = torch.cat([w * torch.eye(4, dtype=torch.float64) for w in weights[b]], 1)
        expected = transform @ dense_covariance(joint, b) @ transform.T
        assert torch.allclose(torch.diag(diag[b]) + factor[b] @ factor[b].T, expected, atol=1e-10)
        assert torch.allclose(mean[b], transform @ joint.mean[b].flatten())


def test_two_level_woodbury_matches_dense_across_compounds_and_missingness():
    model = new_model().double().eval()
    batch = {k: v.double() if v.is_floating_point() else v for k, v in make_batch(b=2).items()}
    distribution = model(batch)
    target = torch.randn_like(distribution.mean)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 1, 2:5] = False
    mask[1, 2, :] = False
    target[~mask] = float("nan")
    full = dense_full_covariance(distribution)
    observed = mask.flatten()
    dense = torch.distributions.MultivariateNormal(distribution.mean.flatten()[observed],
                                                   covariance_matrix=full[observed][:, observed])
    expected = dense.log_prob(target.flatten()[observed])
    actual = distribution.joint_log_prob(target, mask)
    assert torch.allclose(actual, expected, atol=1e-9)
    # Marginal likelihoods remain valid, but their sum is not the joint score.
    assert not torch.allclose(actual, distribution.log_prob(target, mask).sum(), atol=1e-10)
    actual.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_batched_and_blockwise_joint_algebra_values_and_gradients_match():
    model = new_model().double().eval()
    batch = {k: v.double() if v.is_floating_point() else v for k, v in make_batch(b=3).items()}
    distribution = model(batch)
    target = torch.randn_like(distribution.mean)
    mask = torch.rand_like(target) > 0.3
    mask[1] = False
    target[~mask] = float("nan")
    batched = distribution._joint_log_prob_batched(target, mask)
    blockwise = distribution._joint_log_prob_blockwise(target, mask)
    sparse = distribution._joint_log_prob_sparse(target, mask)
    assert torch.allclose(batched, blockwise, atol=1e-10, rtol=1e-12)
    assert torch.allclose(sparse, blockwise, atol=1e-10, rtol=1e-12)
    inputs = (distribution.mean, distribution.diag_var, distribution.local_factors,
               *distribution.environment_loadings)
    gradients1 = torch.autograd.grad(batched, inputs, retain_graph=True)
    gradients2 = torch.autograd.grad(blockwise, inputs, retain_graph=True)
    gradients3 = torch.autograd.grad(sparse, inputs)
    for left, right in zip(gradients1, gradients2):
        assert torch.allclose(left, right, atol=1e-10, rtol=1e-9)
    for left, right in zip(gradients3, gradients2):
        assert torch.allclose(left, right, atol=1e-10, rtol=1e-9)


def test_shared_batch_sampling_has_cross_compound_covariance():
    model, batch = new_model().eval(), make_batch(b=3)
    with torch.no_grad():
        for head in model.factor_heads:
            head.weight.zero_()
            head.bias.fill_(0.4)
        model.scale_head.weight.zero_()
        model.scale_head.bias.fill_(-5)
        model.residual_head.weight.zero_()
        model.residual_head.bias.zero_()
    # First two compounds share environments; the third source is distinct.
    batch["target_group"][2] += 20
    distribution = model(batch)
    covariance = dense_full_covariance(distribution)
    per_compound = 3 * 7
    assert torch.allclose(covariance[0, per_compound], torch.tensor(0.48), atol=1e-6)
    assert covariance[0, 2 * per_compound] == 0
    with torch.no_grad():
        samples = distribution.sample_joint(50000, torch.Generator().manual_seed(60))
        x = samples[:, :, 0, 0]
        empirical = torch.cov(x.T)
    assert abs(empirical[0, 1] - 0.48) < 0.015
    assert abs(empirical[0, 2]) < 0.015


def test_aggregation_preserves_global_environment_covariance():
    model = new_model().double().eval()
    batch = {k: v.double() if v.is_floating_point() else v for k, v in make_batch(b=2).items()}
    distribution = model(batch)
    weights = torch.tensor([[0.25, 0.5, 0.25], [0.5, -0.5, 0.]], dtype=torch.float64)
    aggregated = distribution.aggregate_wells(weights)
    transforms = [torch.cat([w * torch.eye(7, dtype=torch.float64) for w in row], 1) for row in weights]
    transform = torch.block_diag(*transforms)
    expected = transform @ dense_full_covariance(distribution) @ transform.T
    assert torch.allclose(dense_full_covariance(aggregated), expected, atol=1e-10)
    target = torch.randn_like(aggregated.mean)
    dense = torch.distributions.MultivariateNormal(aggregated.mean.flatten(), covariance_matrix=expected)
    assert torch.allclose(aggregated.joint_log_prob(target), dense.log_prob(target.flatten()), atol=1e-9)


def test_environment_cache_preserves_shared_draws_across_compound_chunks():
    model, batch = new_model().eval(), make_batch(b=3)
    batch["target_group"][2, 1:, 1:] += 10
    original = model(batch)
    # Isolate only the shared environment contribution: local and independent
    # random numbers intentionally have no cross-chunk caching contract.
    original.mean = torch.zeros_like(original.mean)
    original.diag_var = torch.zeros_like(original.diag_var)
    original.local_factors = torch.zeros_like(original.local_factors)
    cache = EnvironmentNoiseCache()
    complete = original.sample_joint(7, environment_noise_cache=cache)
    chunks = []
    for start, end in [(0, 1), (1, 3)]:
        chunk = JointGaussian(original.mean[start:end], original.diag_var[start:end],
                               original.factors[start:end], original.local_factors[start:end],
                               tuple(x[start:end] for x in original.environment_loadings),
                               original.environment_groups[start:end])
        chunks.append(chunk.sample_joint(7, environment_noise_cache=cache))
    assert torch.equal(complete, torch.cat(chunks, dim=1))
    assert ("source", 0) in cache._entries
    assert ("batch", 0, 0) in cache._entries
    assert ("plate", 0, 0, 0) in cache._entries


def test_environment_cache_rejects_incompatible_monte_carlo_configuration():
    cache = EnvironmentNoiseCache()
    key = ("source", 9)
    cache.draw(key, 5, 3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError, match="mismatch"):
        cache.draw(key, 6, 3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError, match="mismatch"):
        cache.draw(key, 5, 4, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError, match="mismatch"):
        cache.draw(key, 5, 3, device=torch.device("cpu"), dtype=torch.float64)
    cache.clear()
    assert not cache._entries


def test_exact_conditioning_matches_dense_mean_covariance_and_likelihood():
    model = new_model().double().eval()
    batch = {k: v.double() if v.is_floating_point() else v for k, v in make_batch(b=2).items()}
    distribution = model(batch)
    mask = torch.zeros_like(distribution.mean, dtype=torch.bool)
    mask[:, 0] = True
    mask[0, 0, 2] = False
    observed_y = torch.randn_like(distribution.mean)
    observed_y[~mask] = float("nan")
    posterior = distribution.condition(observed_y, mask, target_indices=(2, 1))
    covariance = dense_full_covariance(distribution)
    observed = mask.flatten()
    future = torch.arange(distribution.mean.numel()).reshape_as(mask)[:, [2, 1]].reshape(-1)
    mu = distribution.mean.flatten()
    oo = covariance[observed][:, observed]
    fo = covariance[future][:, observed]
    residual = observed_y.flatten()[observed] - mu[observed]
    expected_mean = mu[future] + fo @ torch.linalg.solve(oo, residual)
    prior_future_covariance = covariance[future][:, future]
    expected_covariance = prior_future_covariance - fo @ torch.linalg.solve(oo, fo.T)
    assert torch.allclose(posterior.mean.flatten(), expected_mean, atol=1e-10, rtol=1e-10)
    assert torch.allclose(dense_full_covariance(posterior), expected_covariance, atol=1e-10, rtol=1e-10)
    assert torch.linalg.eigvalsh(prior_future_covariance - expected_covariance).min() > -1e-10
    target = torch.randn_like(posterior.mean)
    dense = torch.distributions.MultivariateNormal(expected_mean, covariance_matrix=expected_covariance)
    assert torch.allclose(posterior.joint_log_prob(target), dense.log_prob(target.flatten()), atol=1e-9)
    assert posterior.sample_joint(4).shape == (4, 2, 2, 7)
    assert "not_biological" in posterior.latent_semantics


def test_exact_conditioning_updates_other_compounds_with_shared_environment():
    model, batch = new_model().double().eval(), make_batch(b=3)
    batch = {k: v.double() if v.is_floating_point() else v for k, v in batch.items()}
    batch["target_group"][2] += 20
    with torch.no_grad():
        for head in model.factor_heads:
            head.weight.zero_()
            head.bias.fill_(0.3)
    distribution = model(batch)
    mask = torch.zeros(distribution.mean.shape[:2], dtype=torch.bool)
    mask[0, 0] = True
    observed_y = distribution.mean.detach().clone()
    observed_y[0, 0] += 1
    posterior = distribution.condition(observed_y, mask, (1, 2))
    change = posterior.mean - distribution.mean[:, [1, 2]]
    assert change[1].abs().max() > 1e-3
    assert torch.allclose(change[2], torch.zeros_like(change[2]), atol=1e-12)
    # Future measured values are irrelevant unless the explicit observed mask
    # reveals them. Replacing all such values cannot change the posterior.
    changed_unobserved = observed_y.clone()
    changed_unobserved[~mask] = 99999
    again = distribution.condition(changed_unobserved, mask, (1, 2))
    assert torch.equal(posterior.mean, again.mean)
    assert torch.equal(posterior.factors, again.factors)


def test_exact_conditioning_without_environment_and_no_observations():
    torch.manual_seed(66)
    prior = JointGaussian(torch.randn(2, 3, 4).double(), torch.rand(2, 3, 4).double() + .5,
                           torch.randn(2, 3, 4, 2).double())
    mask = torch.zeros(2, 3, dtype=torch.bool)
    no_observations = prior.condition(torch.full_like(prior.mean, float("nan")), mask, (1, 2))
    assert torch.allclose(no_observations.mean, prior.mean[:, [1, 2]])
    for b in range(2):
        expected = dense_covariance(prior, b)[4:, 4:]
        assert torch.allclose(dense_covariance(no_observations, b), expected)
    mask[:, 0] = True
    y = torch.randn_like(prior.mean)
    posterior = prior.condition(y, mask, (1, 2))
    for b in range(2):
        cov = dense_covariance(prior, b)
        expected = prior.mean[b, 1:].flatten() + cov[4:, :4] @ torch.linalg.solve(
            cov[:4, :4], y[b, 0] - prior.mean[b, 0])
        assert torch.allclose(posterior.mean[b].flatten(), expected, atol=1e-10)


def test_exact_conditioning_rejects_observation_target_overlap():
    distribution = new_model()(make_batch())
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[:, 0] = True
    with pytest.raises(ValueError, match="overlap"):
        distribution.condition(torch.randn_like(distribution.mean), mask, (0, 2))


@pytest.mark.parametrize("mode", ["measurement", "generic", "mlp"])
def test_complete_model_backward_all_modes(mode):
    batch, model = make_batch(), new_model(kernel_mode=mode)
    targets = torch.randn(3, 3, 7)
    loss = model.loss(batch, targets)
    loss["loss"].backward()
    assert torch.isfinite(loss["loss"])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.profile_encoder.group_nets[0][0].weight.grad.abs().sum() > 0
    assert model.factor_heads[0].weight.grad.abs().sum() > 0
    distribution = model(batch)
    assert distribution.sample_joint(5).shape == (5, 3, 3, 7)
    assert (distribution.marginal_variance > 0).all()


def test_masked_reference_and_context_padding_invariance():
    model, batch = new_model().eval(), make_batch()
    batch["context_mask"][:, 1] = False
    batch["context_reference_mask"][:, 0, 1] = False
    batch["target_reference_mask"][:, :, 2] = False
    original = model(batch)
    modified = {k: v.clone() for k, v in batch.items()}
    modified["context_y"][:, 1] = float("nan")
    modified["context_cond"][:, 1] = float("nan")
    modified["context_reference"][:, 1] = float("nan")
    modified["context_reference"][:, 0, 1] = float("nan")
    modified["target_reference"][:, :, 2] = float("nan")
    later = model(modified)
    assert torch.allclose(original.mean, later.mean)
    assert torch.allclose(original.diag_var, later.diag_var)
    assert torch.allclose(original.factors, later.factors)


def test_context_order_invariance_and_empty_set_prior():
    model, batch = new_model().eval(), make_batch()
    original = model(batch)
    reverse = {k: (v.flip(1) if k.startswith("context_") else v) for k, v in batch.items()}
    actual = model(reverse)
    assert torch.allclose(original.mean, actual.mean, atol=1e-6)
    assert torch.allclose(original.factors, actual.factors, atol=1e-6)
    batch["context_mask"][:] = False
    empty = model(batch)
    assert torch.isfinite(empty.mean).all()


def test_group_identifiers_control_covariance_not_mean_lookup():
    model, batch = new_model().eval(), make_batch()
    with torch.no_grad():
        for head in model.factor_heads:
            head.weight.zero_()
            head.bias.fill_(0.2)
        model.residual_head.weight.zero_()
        model.residual_head.bias.zero_()
    distribution = model(batch)
    covariance = dense_covariance(distribution)
    # R2 conditions the chemical latent variance on the observed context.
    # Two shared environments add .08; a new source shares only that latent.
    latent = .04 * model._state_details(batch)["posterior_var"][0].mean()
    assert torch.allclose(covariance[0, 7], latent + .08, atol=1e-6)
    assert torch.allclose(covariance[0, 14], latent, atol=1e-6)
    renamed = copy.deepcopy(batch)
    renamed["target_group"] = renamed["target_group"] + 100
    renamed["context_group"] = renamed["context_group"] + 100
    another = model(renamed)
    assert torch.equal(distribution.mean, another.mean)
    assert torch.equal(distribution.factors, another.factors)


def test_checkpoint_roundtrip_and_frozen_encoder():
    model, batch = new_model().eval(), make_batch()
    clone = MeasurementWorldModel(**model.config).eval()
    clone.load_state_dict(model.state_dict())
    assert torch.equal(model(batch).mean, clone(batch).mean)
    clone.freeze_encoder()
    assert not any(p.requires_grad for p in clone.profile_encoder.parameters())


def test_target_values_rejected_from_inference_dictionary():
    batch = make_batch()
    batch["target_y"] = torch.randn(3, 3, 7)
    with pytest.raises(ValueError, match="separate"):
        new_model()(batch)


def test_nonpartition_feature_groups_rejected():
    with pytest.raises(ValueError):
        GroupedProfileEncoder({"bad": [0, 0, 2]})
