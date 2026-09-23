"""Synthetic full-budget interface test; no assay files or model training."""
import json

import numpy as np
import torch

from opal2.biology_kernel_evaluation import write_json
from opal2.conditional_joint_error_experiment import observable_forward
from opal2.eu_core_distribution import fit_eu_distribution, predict_eu_distribution
from opal2.eu_core_experiment import SAMPLES, extra_seed_moments, metric_summary
from opal2.empirical_radial_experiment import score
from opal2.gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from opal2.gram_oof_ridge import transform_target


def test_complete_distribution_to_existing_full_budget_score_and_json(tmp_path):
    rng = np.random.default_rng(8144)

    def first_inputs(prefix, n):
        return dict(ids=np.array([f'{prefix}{i}' for i in range(n)]),
            groups=np.array([f'{prefix}_group{i}' for i in range(n)]),
            X=rng.normal(size=(n, 24)) * np.exp(rng.normal(size=(n, 1))),
            chem=np.column_stack((rng.integers(0, 2, size=(n, 512)), np.ones(n))))

    ref, cal, query = first_inputs('r', 32), first_inputs('c', 20), first_inputs('q', 2)
    mean = np.zeros((2, 9))
    query['mean_u'] = mean.copy()
    base = np.eye(9) * .8
    fit = fit_eu_distribution(ref, rng.normal(size=(32, 9)), cal,
        rng.normal(size=(20, 9)), base, .75,
        model_training_ids=['train1', 'train2'], model_training_groups=['train1', 'train2'])
    distribution = predict_eu_distribution(fit, query)
    # Outcomes are constructed only after the fitted law and QUERY predictions.
    y = rng.normal(size=(2, 4, 24))
    y[:, 0] = query['X']
    gram = profiles_to_gram(torch.as_tensor(y, dtype=torch.float64))
    raw = gram_to_coordinates(gram).numpy()
    stats = dict(u_center=np.array([.2, -.1, .3, .1, .2, -.1, -.1, .3, .2]),
        u_scale=np.array([.1, .2, .15, .1, .2, .1, .2, .1, .15]))
    target = transform_target(raw, stats)
    actual, obs, difference, _ = observable_forward(raw)
    np.testing.assert_allclose(actual, gram_gains(gram).numpy()[:, 2], atol=1e-12, rtol=1e-12)
    norm2 = np.square(query['X']).mean(1)
    absolute = np.log1p(difference * norm2[:, None])
    reports = {}
    assert SAMPLES == 100000
    for name in ('GAUSSIAN', 'AMP_EMP_LOCAL'):
        radial = name == 'AMP_EMP_LOCAL'
        scatter = distribution['scatter_u'] if radial else distribution['base_scatter_u']
        law = distribution['law'] if radial else None
        weights = distribution['radial_weights'] if radial else None
        scored = score(mean, scatter, target, stats, actual, obs, absolute, norm2,
                       1634, law=law, weights=weights, samples=SAMPLES)
        assert all(np.isfinite(value).all() for value in scored.values())
        np.testing.assert_array_equal(scored['mahalanobis2'],
            np.square(np.linalg.solve(np.linalg.cholesky(scatter),
                                      (target - mean)[..., None])[..., 0]).sum(1))
        if radial:
            np.testing.assert_allclose(scored['covariance_u'], distribution['covariance_u'])
            # Same full-budget streams verify that decision-only sensitivity
            # uses the same standardized -> native conversion and Gamma.
            moment = extra_seed_moments(mean, scatter, stats, law=law, weights=weights, seed=1634)
            np.testing.assert_allclose(moment['predicted'], scored['predicted'], atol=1e-12, rtol=1e-12)
            np.testing.assert_array_equal(moment['p_null'], scored['p_null'])
        else:
            np.testing.assert_array_equal(scored['covariance_u'], scatter)
        reports[name] = metric_summary(scored, actual)
    write_json(tmp_path / 'integration.json', dict(distribution=fit['report'], law=fit['law'],
        prediction_report=distribution['report'], metrics=reports))
    saved = json.loads((tmp_path / 'integration.json').read_text())
    assert saved['law']['dimension'] == 9
    assert saved['distribution']['query_outcomes_used'] is False
    assert len(saved['metrics']['AMP_EMP_LOCAL']['joint_coverage_by_level']) == 5
    np.testing.assert_array_equal(distribution['mean_u'], mean)
