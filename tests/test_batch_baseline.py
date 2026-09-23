"""Software fixtures, not Cell Painting results or model performance claims."""
import numpy as np
import pytest

from opal2.batch_baseline import BatchConditionalResidualBootstrap


def groups(n, *, reverse_alternating=False):
    context = np.tile(["source", "initial_batch", "initial_plate"], (n, 1))
    targets = np.tile([["source", "B", "plate_B"], ["source", "C", "plate_C"]], (n, 1, 1))
    if reverse_alternating:
        targets[1::2] = targets[1::2, ::-1]
    return context, targets


def fitted(seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(120, 5))
    c, t = groups(len(x), reverse_alternating=True)
    is_b = (t[..., 1] == "B")[..., None]
    y = np.where(is_b, 2 + .7 * x[:, None], -2 + 1.8 * x[:, None])
    y += rng.normal(scale=.05, size=y.shape)
    model = BatchConditionalResidualBootstrap(global_ridge=.1, offset_shrinkage=2, slope_shrinkage=2)
    return model.fit(x, y, c, t, train_ids=[f"train-{i}" for i in range(len(x))]), x, y, c, t


def test_actual_batch_not_role_prediction_and_shrinkage():
    model, x, y, c, t = fitted()
    prediction = model.predict_mean(x, c, t)
    pooled = model.global_intercept + ((x - model.x_mean) / model.x_scale)[:, None] * model.global_slope
    assert np.mean((y - prediction)**2) < .01 * np.mean((y - pooled)**2)
    assert model.group_counts.tolist() == [120, 120]
    assert len(model.group_keys) == 2
    for key in model.group_keys:
        np.testing.assert_allclose(model.residuals[model.training_keys == key].mean(0), 0, atol=1e-12)
    qx = np.repeat(x[:1], 2, axis=0)
    qc, qt = groups(2, reverse_alternating=True)
    means = model.predict(qx, qc, qt)
    np.testing.assert_allclose(means[0], means[1, ::-1], atol=1e-12)
    qt[..., 2] = "different_plate_identity"
    qc[..., 2] = "different_context_plate"
    np.testing.assert_array_equal(means, model.predict_mean(qx, qc, qt))


def test_unseen_pair_explicit_training_pooled_fallback():
    model, x, y, c, t = fitted()
    query_groups = t[:3].copy()
    query_groups[:, :, 1] = "never_observed_batch"
    mean = model.predict_mean(x[:3], c[:3], query_groups)
    pooled = model.global_intercept + ((x[:3] - model.x_mean) / model.x_scale) * model.global_slope
    np.testing.assert_allclose(mean[:, 0], pooled)
    np.testing.assert_allclose(mean[:, 1], pooled)
    diag = model.prediction_diagnostics(c[:3], query_groups)
    assert diag["unseen_pair_count"] == 6
    assert all(diag["pooled_noise_fallback"])
    assert "not zero-shot" in diag["scope"]
    samples = model.sample_joint(x[:3], c[:3], query_groups, 7, seed=8)
    assert samples.shape == (7, 3, 2, 5) and np.isfinite(samples).all()


def test_joint_residual_rows_preserved_and_query_order_equivariant():
    rng = np.random.default_rng(2)
    n = 40
    x = np.zeros((n, 3))
    e = rng.normal(size=(n, 1)) * np.array([[1., 2., -.4]])
    y = np.stack((2 + e, -2 + 2 * e), axis=1)
    c, t = groups(n)
    model = BatchConditionalResidualBootstrap().fit(x, y, c, t)
    s = model.sample_joint(x[:1], c[:1], t[:1], 200, seed=7, dtype=np.float64)
    residual = s[:, 0] - model.predict(x[:1], c[:1], t[:1])[0]
    # Every draw is one actual multi-well training residual, not separately
    # resampled wells or features.
    for draw in residual:
        assert np.any(np.max(np.abs(model.residuals - draw[None]), axis=(1, 2)) < 1e-12)
    assert np.corrcoef(residual[:, 0, 0], residual[:, 1, 0])[0, 1] > .999
    reverse = model.sample_joint(x[:1], c[:1], t[:1, ::-1], 200, seed=7, dtype=np.float64)
    np.testing.assert_allclose(s[:, :, ::-1], reverse, atol=1e-12)


def test_training_only_copy_and_checkpoint_reproduction(tmp_path):
    model, x, y, c, t = fitted()
    means = model.predict(x[:4], c[:4], t[:4])
    samples = model.sample_joint(x[:4], c[:4], t[:4], 9, seed=44)
    y[:] = 1e12  # Fit must not retain mutable target references.
    np.testing.assert_array_equal(means, model.predict(x[:4], c[:4], t[:4]))
    np.testing.assert_array_equal(samples, model.sample_joint(x[:4], c[:4], t[:4], 9, seed=44))
    path = tmp_path / "batch_baseline.npz"
    model.save(path)
    restored = BatchConditionalResidualBootstrap.load(path)
    assert model.metadata() == restored.metadata()
    np.testing.assert_array_equal(means, restored.predict(x[:4], c[:4], t[:4]))
    np.testing.assert_array_equal(samples, restored.sample_joint(x[:4], c[:4], t[:4], 9, seed=44))
    assert not model.metadata()["predictive_calibration_certified"]


def test_missing_targets_use_only_genuine_joint_donors():
    model, x, y, c, t = fitted()
    mask = np.ones(y.shape[:2], bool)
    mask[:100, 1] = False
    y[~mask] = np.nan
    model.fit(x, y, c, t, target_mask=mask)
    diag = model.prediction_diagnostics(c[:1], t[:1])
    assert diag["matched_schedule_donors"] == [20]
    assert diag["available_pooled_joint_donors"] == 20
    samples = model.sample_joint(x[:1], c[:1], t[:1], 10, seed=1)
    assert np.isfinite(samples).all()
    three_targets = np.concatenate((t[:1], t[:1, :1]), axis=1)
    with pytest.raises(ValueError, match="enough jointly observed wells"):
        model.sample_joint(x[:1], c[:1], three_targets, 1)


def test_group_ridge_shrinks_tiny_groups_more():
    rng = np.random.default_rng(11)
    x = np.zeros((51, 2)); c, t = groups(51)
    t[0, 0, 1] = "rare_batch"
    y = np.zeros((51, 2, 2))
    y[0, 0] = 100.
    model = BatchConditionalResidualBootstrap(offset_shrinkage=20).fit(x, y, c, t)
    mean = model.predict(x[:1], c[:1], t[:1])
    assert np.all(mean[0, 0] < 10.)  # One exceptional donor cannot set a +100 group offset.
    diag = model.prediction_diagnostics(c[:1], t[:1])
    assert diag["matched_schedule_donors"] == [1] and diag["pooled_noise_fallback"] == [True]


def test_invalid_schema_and_parameters_rejected():
    with pytest.raises(ValueError, match="strictly positive"):
        BatchConditionalResidualBootstrap(slope_shrinkage=0)
    model, x, y, c, t = fitted()
    with pytest.raises(ValueError, match="source/batch/plate"):
        model.predict(x, c[:, :2], t)
    with pytest.raises(ValueError, match="distinct compounds"):
        model.fit(x, y, c, t, train_ids=["duplicate"] * len(x))
    with pytest.raises(ValueError, match="training feature schema"):
        model.predict(x[:, :3], c, t)
