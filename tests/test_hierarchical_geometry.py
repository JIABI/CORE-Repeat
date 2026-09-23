"""Mechanism tests for the bounded residual estimator, using synthetic arrays."""
import inspect
import json

import numpy as np
import pytest
import torch

from opal2.hierarchical_geometry import RidgeResidualMean, sample_joint_coordinates


def make_model(seed=42, width=7, **kwargs):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    coefficient = rng.normal(size=(width, 9))
    intercept = rng.normal(size=9)
    model = RidgeResidualMean(width, coefficient, intercept, **kwargs)
    inputs = torch.tensor(rng.normal(size=(11, width)), dtype=torch.float64)
    return model, inputs, coefficient, intercept


def test_zero_initialization_is_exact_fixed_ridge_and_state_roundtrips():
    model, x, coefficient, intercept = make_model()
    expected = x @ model.coefficient + model.intercept
    for training in (True, False):
        model.train(training)
        assert torch.equal(model(x), expected)
        assert torch.count_nonzero(model.correction(x)) == 0
    assert set(dict(model.named_buffers())) == {"coefficient", "intercept"}
    assert "coefficient" not in dict(model.named_parameters())
    coefficient.fill(88)
    intercept.fill(88)
    assert torch.equal(model.base_mean(x), expected)
    config = json.loads(json.dumps(model.config))
    state = model.state_dict()
    restored = RidgeResidualMean.from_config(config, coefficient=state["coefficient"],
                                            intercept=state["intercept"])
    restored.load_state_dict(state)
    restored.eval()
    assert torch.equal(restored(x), model(x))


def test_only_residual_parameters_receive_gradients_and_step_preserves_base():
    model, x, _, _ = make_model(dropout=0.)
    fixed = model.base_mean(x).clone()
    target = fixed + .2
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    result = model.loss(x, target)
    assert torch.allclose(result["loss"], result["mean_mse"] + result["residual_penalty"])
    assert result["correction_mse"].item() == 0
    result["loss"].backward()
    assert model.coefficient.grad is None and model.intercept.grad is None
    assert model.network[-1].weight.grad.abs().sum() > 0
    # Zero final weights intentionally give earlier layers zero gradients on
    # the first step, not on all subsequent steps.
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.loss(x, target)["loss"].backward()
    assert model.network[0].weight.grad.abs().sum() > 0
    assert torch.equal(model.base_mean(x), fixed)
    assert model.correction(x).abs().max() <= .5


def test_correction_bound_penalty_and_complete_input_are_explicit():
    model, x, _, _ = make_model(dropout=0.)
    with torch.no_grad():
        model.network[-1].bias.copy_(torch.linspace(-100, 100, 9))
    correction = model.correction(x)
    assert correction.abs().max() <= .5
    target = model.base_mean(x)
    result = model.loss(x, target)
    assert torch.allclose(result["correction_mse"], correction.square().mean())
    assert torch.allclose(result["residual_penalty"], .1 * correction.square().mean())
    assert torch.allclose(result["mean_mse"], correction.square().mean())
    assert tuple(inspect.signature(model.forward).parameters) == ("inputs",)
    assert tuple(inspect.signature(model.correction).parameters) == ("inputs",)
    before = model(x).clone()
    target.add_(100)
    assert torch.equal(before, model(x))
    with pytest.raises(ValueError, match="every fitted coordinate"):
        model(x[:, :-1])
    bad = x.clone(); bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(bad)
    with pytest.raises(ValueError, match="dtype"):
        model(x.float())


@pytest.mark.parametrize("object_specific", [False, True])
def test_sampling_recovers_full_joint_covariance_and_is_seeded(object_specific):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(9, 9)) / 3
    covariance = a @ a.T + np.eye(9) * .2
    mean = rng.normal(size=(2, 9))
    if object_specific:
        covariance = np.stack((covariance, covariance * 1.4))
    draws = sample_joint_coordinates(mean, covariance, 30000, 8)
    assert draws.shape == (30000, 2, 9) and draws.dtype == np.float64
    assert np.isfinite(draws).all()
    np.testing.assert_array_equal(draws[:4], sample_joint_coordinates(mean, covariance, 4, 8))
    assert not np.array_equal(draws[:4], sample_joint_coordinates(mean, covariance, 4, 9))
    assert not np.array_equal(draws[:, 0] - mean[0], draws[:, 1] - mean[1])
    np.testing.assert_allclose(draws.mean(0), mean, atol=.03, rtol=0)
    for i in range(2):
        expected = covariance[i] if object_specific else covariance
        np.testing.assert_allclose(np.cov(draws[:, i].T), expected, atol=.04, rtol=.07)


def test_invalid_covariance_and_counts_stop_without_numerical_repair():
    mean = np.zeros((2, 9))
    with pytest.raises(ValueError, match="positive definite"):
        sample_joint_coordinates(mean, np.zeros((9, 9)), 5, 1)
    with pytest.raises(ValueError, match="positive integer"):
        sample_joint_coordinates(mean, np.eye(9), 0, 1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        sample_joint_coordinates(mean, np.eye(9), 5, -1)
    bad = np.eye(9); bad[0, 1] = .5
    with pytest.raises(ValueError, match="symmetric"):
        sample_joint_coordinates(mean, bad, 5, 1)
    with pytest.raises(ValueError, match="shape"):
        sample_joint_coordinates(mean, np.eye(8), 5, 1)
