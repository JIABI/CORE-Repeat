"""Numerical fixtures test isolation and implementation, not biological efficacy."""
import numpy as np
import pytest
import torch

from opal2.conditional_state_representation import (
    ConditionalStateRepresentation, fit_representation, representation_reference_weights)


def fixture_data():
    rng = np.random.default_rng(7)
    state = rng.normal(size=(24, 7))
    profiles = state[:, None]+rng.normal(scale=.15, size=(24, 4, 7))
    profiles[:, 1] += .3
    profiles[:, 2] -= .2
    return profiles, np.repeat(np.arange(12), 2).astype(str), np.array([f"id{i}" for i in range(24)])


def fit_fixture(kind="conditional_predictive", **kwargs):
    profiles, groups, ids = fixture_data()
    return fit_representation(profiles, groups, ids, kind=kind, epochs=3, hidden_dim=8,
                              input_dim=5, target_dim=4, latent_dim=3, batch_size=8, **kwargs)


@pytest.mark.parametrize("kind", ["conditional_predictive", "direct"])
def test_real_training_inference_and_fixed_target(kind):
    model = fit_fixture(kind)
    profiles, _, ids = fixture_data()
    assert model.transform(profiles[:, 0]).shape == (24, 4)
    assert model.predict_targets(profiles[:, 0]).shape == (24, 3, 5)
    assert np.isfinite(model.transform(profiles[:, 0])).all()
    assert len(model.report["history"]) == 3
    assert model.report["best_epoch"] in (1, 2, 3)
    assert model.report["parameter_count"] > 0
    assert model.report["validation_role_mean_baseline_loss"] > 0
    assert not set(model.report["inner_fit_groups"]) & set(model.report["inner_validation_groups"])
    assert not set(model.report["inner_fit_ids"]) & set(model.report["inner_validation_ids"])
    assert all(not parameter.requires_grad for parameter in model.network.parameters())
    assert "teacher" not in dict(model.network.named_modules())
    described = model.describe(profiles[:2, 0], ids[:2])
    assert len(described["records"]) == 2
    assert np.array_equal(described["embedding"], model.transform(profiles[:2, 0]))


def test_preprocessing_is_fit_only_and_future_cannot_enter_transform():
    profiles, groups, ids = fixture_data()
    first = fit_fixture()
    valid = np.isin(ids, first.report["inner_validation_ids"])
    changed = profiles.copy()
    changed[valid] = changed[valid]*100+3
    second = fit_representation(changed, groups, ids, epochs=3, hidden_dim=8,
                               input_dim=5, target_dim=4, latent_dim=3, batch_size=8)
    for name in vars(first.input_coordinates):
        assert np.array_equal(getattr(first.input_coordinates, name), getattr(second.input_coordinates, name))
        assert np.array_equal(getattr(first.target_coordinates, name), getattr(second.target_coordinates, name))
    old = first.transform(profiles[:, 0])
    profiles[:, 1:] = np.nan
    assert np.array_equal(old, first.transform(profiles[:, 0]))
    with pytest.raises(ValueError):
        first.transform(profiles)


def test_explicit_amplitude_bypass_and_direction_are_separate():
    model = fit_fixture()
    X = fixture_data()[0][:3, 0]
    one, two = model.input_coordinates.transform(X), model.input_coordinates.transform(2*X)
    assert np.allclose(one[:, :-1], two[:, :-1], atol=1e-12)
    assert np.allclose(two[:, -1]-one[:, -1], np.log(2)/model.input_coordinates.amp_scale)
    assert np.allclose(model.transform(X)[:, -1], one[:, -1], atol=1e-6)


def test_save_load_and_rng_preserved(tmp_path):
    torch.manual_seed(511)
    before = torch.get_rng_state().clone()
    model = fit_fixture()
    assert torch.equal(before, torch.get_rng_state())
    path = tmp_path/"representation.pt"
    model.save(path)
    loaded = ConditionalStateRepresentation.load(path)
    assert torch.equal(before, torch.get_rng_state())
    X = fixture_data()[0][:, 0]
    assert np.array_equal(model.transform(X), loaded.transform(X))
    assert np.array_equal(model.predict_targets(X), loaded.predict_targets(X))
    assert loaded.report == model.report


def test_deterministic_seed_and_common_input_target_comparator():
    one, two = fit_fixture(), fit_fixture()
    direct = fit_fixture("direct")
    assert one.report == two.report
    assert all(torch.equal(a, b) for a, b in zip(one.network.parameters(), two.network.parameters()))
    assert one.report["inner_fit_ids"] == direct.report["inner_fit_ids"]
    assert np.array_equal(one.target_coordinates.components, direct.target_coordinates.components)
    rates = [item["learning_rate"] for item in one.report["history"]]
    assert rates[0] > rates[-1]


def test_early_stop_restores_best_checkpoint():
    profiles, groups, ids = fixture_data()
    kwargs = dict(hidden_dim=8, input_dim=5, target_dim=4, latent_dim=3, batch_size=8,
                  patience=1, min_delta=1e6)
    model = fit_representation(profiles, groups, ids, epochs=8, **kwargs)
    assert model.report["early_stopped"]
    assert model.report["epochs_completed"] == 2
    assert model.report["best_epoch"] == 1
    prediction = model.predict_targets(profiles[:, 0])
    target = model.target_coordinates.transform(profiles[:, 1:].reshape(-1, 7)).reshape(24, 3, -1)
    valid = np.isin(ids, model.report["inner_validation_ids"])
    errors = (prediction[valid]-target[valid])**2
    loss = .5*(errors[..., :-1].mean()+errors[..., -1].mean())
    assert loss == pytest.approx(model.report["best_validation_loss"], rel=1e-6)


def test_reference_weights_only_embedding_and_group_exclusion():
    references = np.array([[0., 0.], [1., 1.], [2., 1.]])
    query = np.array([[0., 0.], [1., 1.]])
    answer = representation_reference_weights(references, query, bandwidth=.8,
        reference_ids=["a", "b", "c"], query_ids=["a", "q"],
        reference_groups=["g1", "g2", "g3"], query_groups=["g1", "g2"])
    assert np.allclose(answer["weights"].sum(1), 1.)
    assert answer["weights"][0, 0] == 0.
    assert answer["weights"][1, 1] == 0.
    assert np.all(answer["ess"] >= answer["local_ess"]-1e-12)
    with pytest.raises(ValueError, match="eligible"):
        representation_reference_weights(references[:1], query[:1], bandwidth=1.,
                                          reference_ids=["a"], query_ids=["a"])


def test_common_weight_interface_and_bad_inputs():
    model = fit_fixture()
    X = fixture_data()[0][:, 0]
    weights = model.reference_weights(X[:10], X[10:])
    assert weights["weights"].shape == (14, 10)
    assert np.allclose(weights["weights"].sum(1), 1.)
    profiles, groups, ids = fixture_data()
    with pytest.raises(ValueError, match="60"):
        fit_representation(profiles, groups, ids, epochs=61)
    with pytest.raises(ValueError, match="unique"):
        fit_representation(profiles, groups, ["x"]*24)
    with pytest.raises(ValueError, match="finite"):
        fit_representation(profiles*np.nan, groups, ids)
