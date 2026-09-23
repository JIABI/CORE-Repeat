"""Synthetic engineering tests, not empirical biological evidence."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from opal2.closed_form_baseline import ClosedFormBaseline, fit_baseline
from opal2.data import TrainScaler


def fixture(seed=15, n=60, w=4, d=8):
    rng = np.random.default_rng(seed)
    loading = rng.normal(size=(3, d))
    signal = rng.normal(size=(n, 3)) @ loading
    means = rng.normal(size=(w, d))
    scale = np.linspace(.3, 2., d)
    y = (signal[:, None] + means[None] + rng.normal(size=(n, w, d))) * scale
    return y


def dense_joint(model, slots):
    common = model._signal_factor @ model._signal_factor.T
    noise = model._noise_factor @ model._noise_factor.T + np.diag(model.residual_var)
    t = len(slots)
    covariance = np.kron(np.ones((t, t)), common) + np.kron(np.eye(t), noise)
    scale = np.tile(model.scale, t)
    covariance *= scale[:, None] * scale[None]
    mean = (model.center + model.slot_mean[slots] * model.scale).ravel()
    return mean, covariance


def test_affine_matches_existing_scaler_and_exact_clipped_slot_mean(tmp_path):
    y = fixture()
    y[0, 1, 0] = 10000.
    model = fit_baseline(y, k=4, clip=3.)
    center, scale = TrainScaler._moments(y.reshape(-1, y.shape[-1]), 1e-6)
    np.testing.assert_array_equal(model.center, center)
    np.testing.assert_array_equal(model.scale, scale)
    robust = np.clip((y - center) / scale, -3, 3)
    np.testing.assert_array_equal(model.slot_mean, robust.mean(axis=0))
    np.testing.assert_allclose(model.transform_features(y, [0, 1, 2, 3], clip=True).mean(axis=0),
                               0, atol=1e-14)
    path = tmp_path / "fitted_model.npz"
    model.save(path)
    loaded = ClosedFormBaseline.load(path)
    np.testing.assert_array_equal(loaded.slot_mean, model.slot_mean)
    np.testing.assert_array_equal(loaded.center, model.center)
    before = model.conditional(y[:2, :1], [0], [1, 2, 3])
    after = loaded.conditional(y[:2, :1], [0], [1, 2, 3])
    np.testing.assert_allclose(before.mean, after.mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(before.log_prob(y[:2, 1:]), after.log_prob(y[:2, 1:]), atol=1e-12)
    np.testing.assert_allclose(before.sample_joint(3, seed=3), after.sample_joint(3, seed=3), atol=1e-12)


@pytest.mark.parametrize("context_slots,target_slots", [([0], [1, 2, 3]), ([0, 2], [1, 3]), ([], [1, 3])])
def test_full_gaussian_conditioning_and_joint_log_density_equal_dense(context_slots, target_slots):
    y = fixture()
    model = fit_baseline(y, k=3)
    inputs = y[:3][:, context_slots]
    prediction = model.conditional(inputs, context_slots, target_slots)
    slots = context_slots + target_slots
    mean, covariance = dense_joint(model, slots)
    c = len(context_slots) * y.shape[-1]
    if c:
        cc = covariance[:c, :c]
        tc = covariance[c:, :c]
        conditional_mean = mean[c:] + (inputs.reshape(3, -1) - mean[:c]) @ np.linalg.solve(cc, tc.T)
        conditional_covariance = covariance[c:, c:] - tc @ np.linalg.solve(cc, tc.T)
    else:
        conditional_mean = np.broadcast_to(mean, (3, len(mean)))
        conditional_covariance = covariance
    np.testing.assert_allclose(prediction.mean.reshape(3, -1), conditional_mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(prediction.dense_covariance(), conditional_covariance, rtol=1e-9, atol=1e-9)
    target = y[:3][:, target_slots]
    expected = [multivariate_normal.logpdf(target[i].ravel(), conditional_mean[i], conditional_covariance)
                for i in range(3)]
    np.testing.assert_allclose(prediction.log_prob(target), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(prediction.nll(target), -np.asarray(expected) / target.shape[1] / target.shape[2])


def test_positive_full_space_residual_and_legal_covariances():
    y = fixture(d=14)
    model = fit_baseline(y, k=2)
    assert np.linalg.eigvalsh(model.signal_cov).min() >= -1e-10
    assert np.linalg.eigvalsh(model.within_cov).min() > 0
    assert model.residual_var.min() > 0
    prediction = model.conditional(y[:1, :1], [0], [1, 2])
    covariance = prediction.dense_covariance()
    assert np.linalg.eigvalsh(covariance).min() > 0
    _, _, vt = np.linalg.svd(model.basis.T, full_matrices=True)
    outside = vt[-1] / model.scale
    # A discarded PCA direction has real stochastic residual variance.
    vec = np.concatenate((outside, np.zeros_like(outside)))
    assert vec @ covariance @ vec > 1e-5
    draw = prediction.sample_joint(12000, seed=99).reshape(12000, -1)
    np.testing.assert_allclose(np.var(draw @ vec), vec @ covariance @ vec, rtol=.06)


def test_conditioning_does_not_winsorise_new_inputs():
    y = fixture()
    model = fit_baseline(y, k=4, clip=1.)
    x = model.center + model.scale * 30
    inputs = np.stack((x, x + model.scale * 20))[:, None]
    assert np.allclose(model.transform_features(inputs, [0], clip=True)[0],
                       model.transform_features(inputs, [0], clip=True)[1])
    assert not np.allclose(model.conditional(inputs, [0], [1]).mean[0],
                           model.conditional(inputs, [0], [1]).mean[1])


def test_reliability_basis_whitens_noise_and_orders_signal():
    model = fit_baseline(fixture(), k=5)
    q = model.within_cov + model.basis.T @ (model.residual_var[:, None] * model.basis)
    vectors = model.reliability_vectors
    np.testing.assert_allclose(vectors.T @ q @ vectors, np.eye(5), atol=1e-8)
    np.testing.assert_allclose(vectors.T @ model.signal_cov @ vectors,
                               np.diag(model.reliability_eigenvalues), atol=1e-8)
    assert np.all(np.diff(model.reliability_eigenvalues) <= 0)
    x = fixture()[:3, 0]
    np.testing.assert_allclose(model.reliability_features(x),
                               model.transform_features(x, 0, clip=False) @ model.basis @ vectors)


def test_more_information_reduces_same_target_variance():
    y = fixture()
    model = fit_baseline(y, k=4)
    one = model.conditional(y[:2, :1], [0], [3])
    two = model.conditional(y[:2, :2], [0, 1], [3])
    assert np.linalg.eigvalsh(one.dense_covariance() - two.dense_covariance()).min() > -1e-9


def test_targets_share_signal_in_monte_carlo():
    y = fixture(d=4)
    model = fit_baseline(y, k=2)
    prediction = model.conditional(y[:1, :1], [0], [1, 2])
    draw = prediction.sample_joint(18000, seed=302).reshape(18000, -1)
    actual_cov = np.cov(draw, rowvar=False)
    expected_cov = prediction.dense_covariance()
    assert np.linalg.norm(expected_cov[:4, 4:]) > .01
    np.testing.assert_allclose(actual_cov[:4, 4:], expected_cov[:4, 4:], atol=.08, rtol=.15)


def test_gamma_prediction_original_space_shared_draws_and_seed():
    y = fixture(n=30, d=4)
    model = fit_baseline(y[:20], k=3)
    x = y[20:23, 0]
    a = model.gamma_predict(x, n_samples=32, seed=33, chunk_size=2)
    b = model.gamma_predict(x, n_samples=32, seed=33, chunk_size=2)
    for name in a:
        assert a[name].shape == (3, 3)
        np.testing.assert_array_equal(a[name], b[name])
    np.testing.assert_allclose(a["p_null"] + a["p_positive"] + a["p_ambiguous"], 1.)
    np.testing.assert_allclose(a["mc_se"], a["sd"] / np.sqrt(32))
    assert np.all(a["mean"] <= 1) and np.all(a["mean"] >= -1.02)
    before = model.center.copy(), model.scale.copy(), model.slot_mean.copy()
    model.gamma_predict(x * 100, n_samples=4)
    for original, now in zip(before, (model.center, model.scale, model.slot_mean)):
        np.testing.assert_array_equal(original, now)


def test_gamma_predict_retains_original_positive_margin_and_ambiguous_state(monkeypatch):
    model = fit_baseline(fixture(n=20, d=2), k=2)
    gains = np.array([-.002, .002, .006, .03])
    cosine = 2 * (gains + .01)
    addition_first_coordinate = cosine / np.sqrt(1 - cosine**2)
    draws = np.zeros((4, 1, 2, 2))
    draws[:, 0, 0, 0] = addition_first_coordinate
    draws[:, 0, 1, 0] = 1.  # Fixed V=(1,0); X=(0,1).

    class FixedDraws:
        def sample_joint(self, n_samples, rng=None):
            assert n_samples == 4
            return draws

    monkeypatch.setattr(model, "conditional", lambda *args, **kwargs: FixedDraws())
    result = model.gamma_predict(np.array([[0., 1.]]), actions=((1,),), n_samples=4)
    np.testing.assert_allclose(result["mean"], [[gains.mean()]])
    np.testing.assert_array_equal(result["p_null"], [[.25]])
    np.testing.assert_array_equal(result["p_positive"], [[.5]])
    np.testing.assert_array_equal(result["p_ambiguous"], [[.25]])
    with pytest.raises(ValueError, match="Positive margin"):
        model.gamma_predict(np.array([[0., 1.]]), positive_margin=0)


def test_chunked_samples_are_bounded_and_reproducible():
    y = fixture()
    p = fit_baseline(y, k=3).conditional(y[:2, :1], [0], [1, 3])
    chunks = list(p.iter_sample_chunks(17, chunk_size=6, seed=10))
    assert [len(x) for x in chunks] == [6, 6, 5]
    other = list(p.iter_sample_chunks(17, chunk_size=6, seed=10))
    np.testing.assert_array_equal(np.concatenate(chunks), np.concatenate(other))


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"clip": 0}, {"noise_shrinkage": 1.1}, {"variance_floor": 0}])
def test_bad_fit_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_baseline(fixture(), **kwargs)


def test_invalid_or_reused_physical_slots_rejected():
    y = fixture()
    m = fit_baseline(y, k=3)
    for cs, ts, inputs in [([0], [0], y[:2, :1]), ([0, 0], [1], y[:2, :2]),
                           ([0], [4], y[:2, :1]), ([0.5], [1], y[:2, :1])]:
        with pytest.raises(ValueError):
            m.conditional(inputs, cs, ts)
    with pytest.raises(ValueError):
        m.gamma_predict(y[:2, 0], actions=((1, 3),))


def test_constant_coordinates_and_no_robust_clip_remain_finite():
    y = fixture(d=5)
    y[:, :, -1] = 8.
    m = fit_baseline(y, k=5, clip=None)
    assert m.scale[-1] == 1
    p = m.conditional(y[:2, :1], [0], [1, 2, 3])
    assert np.isfinite(p.log_prob(y[:2, 1:])).all()
    assert np.isfinite(p.sample_joint(3, seed=2)).all()
