import numpy as np
import pytest

from opal2.baseline_experiment import actual_gains, forecast, probe_metrics, paired_probe_report
from opal2.closed_form_baseline import fit_baseline


def test_original_endpoint_and_costs():
    y = np.array([[[1., 0.], [0., 1.], [1., 0.], [0., 1.]]])
    gain = actual_gains(y)[0]
    np.testing.assert_allclose(gain, [.5 / np.sqrt(2) - .01, -.01,
                                    .5 / np.sqrt(5) - .02])


def test_forecast_full_space_joint_density_and_distributions():
    rng = np.random.default_rng(9)
    y = rng.normal(size=(20, 4, 8)) + rng.normal(size=(20, 1, 8))
    model = fit_baseline(y[:15], k=3)
    result = forecast(model, y[15:], samples=64, seed=8, object_chunk=2, mc_chunk=16)
    assert result["prediction_mean"].shape == (5, 3, 8)
    assert np.isfinite(result["standardized_nll"]).all()
    assert np.all(result["p_null"] + result["p_positive"] <= 1.)
    assert np.all(result["predictive_sd"] > 0.)
    np.testing.assert_allclose(result["mc_se"], result["predictive_sd"] / 8)
    cond = model.conditional(y[15:, :1], [0], [1, 2, 3])
    np.testing.assert_allclose(result["raw_nll"], -cond.log_prob(y[15:, 1:]) / 24)


def test_training_mean_skill_is_not_standard_r2():
    target = np.arange(24.).reshape(4, 3, 2)
    prediction = target + 1.
    metrics = probe_metrics(target, prediction, np.zeros((3, 2)), np.ones(2))
    assert metrics["r2_evaluation_mean"] != metrics["standardized"]["overall"]["r2_training_mean"]
    report = paired_probe_report({"a": metrics, "b": metrics}, seed=1, n_bootstrap=20)
    assert report[0]["percentile95_interval"] == [0., 0.]


def test_zero_norm_is_not_a_silent_removed_object():
    with pytest.raises(ValueError, match="Zero-norm"):
        actual_gains(np.zeros((1, 4, 2)))
