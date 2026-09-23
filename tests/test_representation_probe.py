"""Matched physical-target representation diagnostics on synthetic fixtures."""
import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from opal2.data import TrainScaler
from opal2.model import GroupedProfileEncoder
from opal2.representation_probe import (extract_checkpoint_features, fit_ridge_probe,
                                        regression_metrics, _fit_head)


def fixture():
    rng = np.random.default_rng(83)
    x = rng.normal(size=(36, 7))
    beta = rng.normal(size=(7, 3, 11))
    y = np.einsum("np,psd->nsd", x, beta) + rng.normal(scale=.08, size=(36, 3, 11))
    y += np.arange(3)[None, :, None] * 12
    return x, y, np.arange(24), np.arange(24, 36)


def test_full_coordinate_multi_slot_prediction_and_common_baseline():
    x, y, train, test = fixture()
    result = fit_ridge_probe(x, y, train, test)
    assert result["predictions"].shape == (12, 3, 11)
    assert result["standardized_predictions"].shape == (12, 3, 11)
    assert result["selected_alphas"].shape == (3,)
    assert result["cv_standardized_mse"].shape == (5, 3)
    assert result["physical_metrics"]["overall"]["r2_training_mean"] > .9
    model = result["model"]
    np.testing.assert_allclose(model.target_center, y[train].mean(0))
    baseline = ((y[test] - y[train].mean(0)) ** 2).sum(axis=(1, 2))
    np.testing.assert_allclose(result["physical_metrics"]["per_object_baseline_sse"], baseline)
    assert result["physical_metrics"]["per_object_slot_sse"].shape == (12, 3)


def test_heldout_targets_cannot_change_selection_transform_or_prediction():
    x, y, train, test = fixture()
    first = fit_ridge_probe(x, y, train, test)
    changed = y.copy()
    changed[test] = 1e8
    second = fit_ridge_probe(x, changed, train, test)
    for field in ("predictions", "selected_alphas", "cv_standardized_mse"):
        np.testing.assert_array_equal(first[field], second[field])
    np.testing.assert_array_equal(first["model"].target_scale, second["model"].target_scale)
    assert first["physical_metrics"]["overall"]["sse"] != second["physical_metrics"]["overall"]["sse"]


def test_unselected_rows_are_not_read_and_missing_selected_rows_are_not_dropped():
    x, y, train, test = fixture()
    x[-1] = np.nan
    y[-1] = np.nan
    fit_ridge_probe(x, y, train, test[:-1])
    with pytest.raises(ValueError, match="no implicit row removal"):
        fit_ridge_probe(x, y, train, test)


def test_alpha_uses_train_standardized_dimension_normalized_kernel():
    x, y, train, test = fixture()
    a = fit_ridge_probe(x, y, train, test)
    b = fit_ridge_probe(x * np.arange(1, 8) + 100, y, train, test)
    c = fit_ridge_probe(np.tile(x, (1, 3)), y, train, test)
    constant = fit_ridge_probe(np.column_stack((x, np.ones(len(x)) * 42)), y, train, test)
    for result in (b, c, constant):
        np.testing.assert_allclose(result["predictions"], a["predictions"], atol=1e-9)
        np.testing.assert_array_equal(result["selected_alphas"], a["selected_alphas"])
    z = a["model"].normalized_training_features
    assert np.mean(np.sum(z*z, axis=1)) == pytest.approx(1)
    assert constant["active_input_dimension"] == 7


def test_target_affine_units_do_not_change_standardized_choice():
    x, y, train, test = fixture()
    scales = np.arange(1, 12)[None, None, :]
    original = fit_ridge_probe(x, y, train, test)
    changed = fit_ridge_probe(x, y * scales + 25, train, test)
    np.testing.assert_array_equal(original["selected_alphas"], changed["selected_alphas"])
    np.testing.assert_allclose(changed["predictions"], original["predictions"] * scales + 25)
    np.testing.assert_allclose(changed["standardized_predictions"], original["standardized_predictions"], atol=1e-10)


def test_inner_fold_feature_builder_refits_without_outer_test_or_inner_test():
    x, y, train, test = fixture()
    calls = []
    def builder(fitted, query):
        calls.append((fitted.copy(), query.copy()))
        assert set(fitted).issubset(train)
        center = x[fitted].mean(0)
        return x[query] - center
    result = fit_ridge_probe(None, y, train, test, feature_builder=builder)
    assert result["representation_fit"] == "inner_fold_refit"
    assert len(calls) == 4
    for (fitted, query), fold in zip(calls[:3], result["inner_folds"]):
        np.testing.assert_array_equal(fitted, fold["train_indices"])
        assert not set(fitted).intersection(fold["validation_indices"])
        assert set(query) == set(train)
    assert set(calls[-1][0]) == set(train)
    assert set(calls[-1][1]) == set(train) | set(test)


def test_dual_probe_equals_explicit_primal_ridge_for_single_slot():
    x, y, train, test = fixture()
    y = y[:, 0]
    alpha = .01
    model = _fit_head(x[train], y[train], alpha, squeezed=True)
    z = model.normalized_training_features
    t = (y[train] - model.target_center[0]) / model.target_scale[0]
    beta = np.linalg.solve(z.T @ z + len(train)*alpha*np.eye(z.shape[1]), z.T @ t)
    query = (x[test] - model.feature_center) / model.feature_scale / np.sqrt(x.shape[1])
    expected = (query @ beta) * model.target_scale[0] + model.target_center[0]
    np.testing.assert_allclose(model.predict(x[test]), expected, atol=1e-10)
    result = fit_ridge_probe(x, y, train, test)
    assert result["predictions"].shape == (12, 11)


def test_metrics_keep_full_space_errors_and_undefined_r2_explicit():
    target = np.ones((3, 4))
    prediction = target + 2
    metrics = regression_metrics(target, prediction, np.ones(4))
    assert metrics["overall"]["sse"] == 48
    assert metrics["overall"]["mse"] == 4
    assert metrics["overall"]["r2_training_mean"] is None
    scaled = regression_metrics(target, prediction, np.zeros(4), np.ones(4)*2)
    assert scaled["overall"]["mse"] == 1
    assert scaled["overall"]["r2_training_mean"] == -3


@pytest.mark.parametrize("kwargs", [dict(alphas=(.1, .1)), dict(alphas=(0.,)),
                                   dict(inner_splits=1), dict(minimum_scale=0)])
def test_invalid_tuning_contract_rejected(kwargs):
    x, y, train, test = fixture()
    with pytest.raises(ValueError):
        fit_ridge_probe(x, y, train, test, **kwargs)
    with pytest.raises(ValueError, match="overlap"):
        fit_ridge_probe(x, y, train, np.array([0, 24]))


def checkpoint_fixture(tmp_path):
    torch.manual_seed(97)
    names = ["Cells_0", "Cells_1", "Nuclei_0", "Nuclei_1"]
    groups = {"Cells": [0, 1], "Nuclei": [2, 3]}
    config = dict(feature_groups=groups, hidden_dim=8, group_attention_layers=1, attention_heads=2)
    encoder = GroupedProfileEncoder(groups, 8, 1, 2).eval()
    projector = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8)).eval()
    state = {"teacher_encoder." + k: v for k, v in encoder.state_dict().items()}
    state.update({"teacher_projector." + k: v for k, v in projector.state_dict().items()})
    manifest = dict(model_config=config, train_config=dict(hidden_dim=8),
                    train_ids=["a", "b", "c"], validation_ids=["d"], feature_names=names)
    scaler = TrainScaler(np.arange(4), np.arange(1, 5), np.zeros(1), np.ones(1),
                         np.zeros((3, 1)), np.ones((3, 1)), manifest["train_ids"], names)
    scaler.save(tmp_path / "scaler.json")
    (tmp_path / "training_diagnostic_manifest.json").write_text(json.dumps(manifest))
    payload = dict(manifest=manifest, state_dict=state,
                   encoder_state_dict=copy.deepcopy(encoder.state_dict()), arm="fixture", epoch=60)
    torch.save(payload, tmp_path / "epoch_060.pt")
    return names, scaler, encoder, projector, payload


def test_checkpoint_extraction_strict_scaler_and_teacher_match_without_rng_changes(tmp_path):
    names, scaler, encoder, projector, _ = checkpoint_fixture(tmp_path)
    y = np.random.default_rng(16).normal(size=(7, 2, 4))
    before = torch.get_rng_state().clone()
    features, metadata = extract_checkpoint_features(y, names, tmp_path, "epoch_060.pt", batch_size=3,
                                                    expected_scaler=scaler)
    assert torch.equal(before, torch.get_rng_state())
    with torch.inference_mode():
        direct = encoder(torch.tensor(scaler.transform_y(y), dtype=torch.float32)).numpy()
    np.testing.assert_allclose(features, direct, atol=2e-6)
    assert features.shape == (7, 2, 8)
    assert metadata["pretraining_ids"] == ["a", "b", "c"]
    assert metadata["optimization_performed"] is False
    assert metadata["numerical_scaler_independently_verified"] is True
    projected, pmeta = extract_checkpoint_features(y, names, tmp_path, "epoch_060.pt", projector=True)
    with torch.inference_mode():
        expected = projector(torch.tensor(direct)).numpy()
    np.testing.assert_allclose(projected, expected, atol=2e-6)
    assert "projector" in pmeta["representation"]


def test_checkpoint_wrong_coordinates_or_export_are_rejected(tmp_path):
    names, _, _, _, payload = checkpoint_fixture(tmp_path)
    y = np.zeros((4, 4))
    with pytest.raises(ValueError, match="coordinate order"):
        extract_checkpoint_features(y, names[::-1], tmp_path, "epoch_060.pt")
    key = next(iter(payload["encoder_state_dict"]))
    payload["encoder_state_dict"][key] = payload["encoder_state_dict"][key] + 1
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="differs from checkpoint"):
        extract_checkpoint_features(y, names, tmp_path, "bad.pt")


def test_checkpoint_pretraining_binding_and_architecture_state_are_strict(tmp_path):
    names, scaler, _, _, payload = checkpoint_fixture(tmp_path)
    scaler.train_ids = ["not-the-pretraining-set"]
    scaler.save(tmp_path / "scaler.json")
    with pytest.raises(ValueError, match="pretraining compound"):
        extract_checkpoint_features(np.zeros((2, 4)), names, tmp_path, "epoch_060.pt")


def test_checkpoint_numeric_scaler_can_be_checked_against_training_refit(tmp_path):
    names, scaler, _, _, _ = checkpoint_fixture(tmp_path)
    refitted = copy.deepcopy(scaler)
    scaler.y_center = scaler.y_center + .01
    scaler.save(tmp_path / "scaler.json")
    with pytest.raises(ValueError, match="independently refitted"):
        extract_checkpoint_features(np.zeros((2, 4)), names, tmp_path, "epoch_060.pt", expected_scaler=refitted)
