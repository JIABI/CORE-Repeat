import numpy as np
import torch

from opal2.gram_geometry import profiles_to_gram, gram_gains
from opal2.gram_evaluation import paired_energy_score, score_grams, fit_score_scale


def test_energy_score_perfect_point_mass_is_zero():
    y = np.array([[1., 2.], [3., 4.]])
    samples = np.broadcast_to(y, (8, 2, 2))
    np.testing.assert_array_equal(paired_energy_score(samples, y), np.zeros(2))


def test_shared_evaluator_preserves_actions_and_endpoint():
    rng = np.random.default_rng(9)
    y = torch.tensor(rng.normal(size=(12, 4, 16)), dtype=torch.float64)
    actual = profiles_to_gram(y).numpy()
    samples = np.broadcast_to(actual, (10, 12, 4, 4)).copy()
    gains = gram_gains(torch.tensor(actual)).numpy()
    report, traces = score_grams(samples, actual, [f"c{i:03}" for i in range(12)],
        train_actual_gains=gains[:6], score_scale=fit_score_scale(actual[:6]),
        n_bootstrap=12, n_random=12)
    np.testing.assert_allclose(traces["predicted"], gains, atol=1e-14)
    np.testing.assert_allclose(traces["utility_crps"], 0, atol=1e-14)
    assert report["formal_certificate"] is False
    assert report["endpoint_changed"] is False
