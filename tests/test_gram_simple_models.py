"""Synthetic mechanics only; no real experiment rows or held-out outcomes."""
import inspect

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from opal2.gram_simple_models import (
    GramSimpleGaussian, REGULARIZATION_GRID, fit_global, fit_ridge, load,
)


@pytest.fixture(autouse=True)
def bounded_test_blas():
    with threadpool_limits(limits=1):
        yield


def synthetic(n=45, p=11, seed=20):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    coefficient = rng.normal(size=(p, 9))
    intercept = rng.normal(size=9)
    shared = rng.normal(size=(n, 1))*.08
    u = x@coefficient+intercept+shared+rng.normal(size=(n, 9))*.04
    return x, u, coefficient, intercept


def test_global_full_joint_covariance_and_shared_draws_do_not_fake_ranking():
    rng = np.random.default_rng(4)
    u = rng.normal(size=(150, 1))*np.linspace(.5, 1.5, 9)+rng.normal(size=(150, 9))*.2
    model = fit_global(u)
    assert np.linalg.eigvalsh(model.covariance).min() > 0
    assert abs(model.covariance[0, 1]) > .1
    x = rng.normal(size=(7, 19))
    prediction = model.predict_mean(x)
    np.testing.assert_array_equal(prediction, np.broadcast_to(u.mean(0), (7, 9)))
    draws = model.sample_coordinates(x, 30000, seed=13)
    assert draws.shape == (30000, 7, 9)
    for j in range(7):
        np.testing.assert_array_equal(draws[:, 0], draws[:, j])
    np.testing.assert_allclose(np.cov(draws[:, 0].T), model.covariance, atol=.035, rtol=.05)
    assert model.metadata["monte_carlo_coupling_is_physical_dependence"] is False


def test_ridge_recovery_regularization_convention_and_complete_input():
    x, u, coefficient, intercept = synthetic(n=85, p=8)
    model = fit_ridge(x, u)
    assert model.selected_lambda in REGULARIZATION_GRID
    assert model.coefficient.shape == (8, 9)
    expected = Ridge(alpha=len(x)*model.selected_lambda, fit_intercept=True).fit(x, u)
    np.testing.assert_allclose(model.coefficient, expected.coef_.T, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(model.intercept, expected.intercept_, rtol=2e-10, atol=2e-10)
    query = np.random.default_rng(30).normal(size=(20, 8))
    truth = query@coefficient+intercept
    mse = np.square(model.predict_mean(query)-truth).mean()
    assert mse < .03
    with pytest.raises(ValueError, match="complete fitted input"):
        model.predict_mean(query[:, :-1])
    assert np.linalg.eigvalsh(model.covariance).min() > 0


def test_nested_oof_exact_once_and_residual_second_moment_keeps_bias():
    x, u, _, _ = synthetic()
    model = fit_ridge(x, u, seed=51)
    audit = model.audit_arrays
    np.testing.assert_array_equal(audit["oof_count"], np.ones(len(x), dtype=int))
    np.testing.assert_allclose(audit["oof_residuals"], u-audit["oof_predictions"])
    np.testing.assert_allclose(audit["residual_mean"], audit["oof_residuals"].mean(0))
    np.testing.assert_allclose(model.covariance, audit["centered_residual_covariance"]+
                               np.outer(audit["residual_mean"], audit["residual_mean"]))
    assert not model.metadata["prediction_bias_correction"]
    for outer in model.metadata["outer_cv"]:
        fit, check = set(outer["outer_fit_indices"]), set(outer["outer_validation_indices"])
        assert fit.isdisjoint(check) and fit | check == set(range(len(x)))
        seen = []
        for inner in outer["inner_folds"]:
            inner_fit, inner_check = set(inner["fit_indices"]), set(inner["validation_indices"])
            assert inner_fit.isdisjoint(inner_check)
            assert inner_fit | inner_check == fit
            assert (inner_fit | inner_check).isdisjoint(check)
            seen.extend(inner["validation_indices"])
        assert sorted(seen) == sorted(fit)
    assert model.metadata["formal_certificate"] is False


def test_one_outer_query_label_does_not_affect_its_oof_prediction():
    x, u, _, _ = synthetic(n=30, p=6)
    original = fit_ridge(x, u, seed=16)
    changed = u.copy()
    changed[2] += 30
    alternative = fit_ridge(x, changed, seed=16)
    # That row may affect other folds, final fitting and the residual covariance,
    # but not its own outer-held prediction or inner penalty selection.
    np.testing.assert_array_equal(original.audit_arrays["oof_predictions"][2],
                                  alternative.audit_arrays["oof_predictions"][2])


@pytest.mark.parametrize("kind", ["GLOBAL", "RIDGE"])
def test_serialization_reproducibility_and_query_target_free_interface(tmp_path, kind):
    x, u, _, _ = synthetic(n=30, p=7)
    model = fit_global(u) if kind=="GLOBAL" else fit_ridge(x, u)
    path = tmp_path / "saved_model.bin"
    model.save(path)
    restored = load(path)
    assert path.is_file() and not path.with_suffix(".bin.npz").exists()
    assert isinstance(restored, GramSimpleGaussian)
    assert restored.metadata == model.metadata
    np.testing.assert_array_equal(restored.predict_mean(x[:5]), model.predict_mean(x[:5]))
    np.testing.assert_array_equal(restored.sample_coordinates(x[:5], 17, 8),
                                  model.sample_coordinates(x[:5], 17, 8))
    for key in model.audit_arrays:
        np.testing.assert_array_equal(restored.audit_arrays[key], model.audit_arrays[key])
    assert tuple(inspect.signature(restored.predict_mean).parameters) == ("x",)
    assert tuple(inspect.signature(restored.sample_coordinates).parameters) == ("x", "samples", "seed")
    before = restored.predict_mean(x[:5])
    for value in restored.audit_arrays.values():
        if np.issubdtype(value.dtype, np.floating):
            value[...] = 999
    np.testing.assert_array_equal(before, restored.predict_mean(x[:5]))
    with pytest.raises(FileExistsError):
        model.save(path)


def test_ridge_supports_more_input_coordinates_than_rows():
    x, u, _, _ = synthetic(n=25, p=55)
    model = fit_ridge(x, u)
    assert model.coefficient.shape == (55, 9)
    assert np.isfinite(model.predict_mean(x[:3])).all()
    assert model.metadata["kernel_eigendecompositions"] == 26


def test_input_errors_and_degenerate_covariance_are_not_hidden():
    with pytest.raises(ValueError, match="positive definite"):
        fit_global(np.ones((12, 9)))
    x, u, _, _ = synthetic(n=10, p=4)
    with pytest.raises(ValueError, match="9 columns"):
        fit_global(u[:, :8])
    with pytest.raises(ValueError, match="same ordered"):
        fit_ridge(x, u[:-1])
    bad = u.copy()
    bad[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_ridge(x, bad)
    model = fit_global(u)
    with pytest.raises(ValueError, match="positive integer"):
        model.sample_coordinates(x, 0, 3)
    with pytest.raises(ValueError, match="nonnegative integer"):
        model.sample_coordinates(x, 5, -1)
