"""Exercise the actual runner calibration route with poisoned query outcomes."""
from copy import deepcopy

import numpy as np

import opal2.variance_headroom_math as math
from opal2.variance_headroom_experiment import calibration_search


def test_actual_calibration_search_excludes_query_futures_and_own_radial_group(monkeypatch):
    rng = np.random.default_rng(913)
    prior = dict(mean_u=rng.normal(size=(6, 9)), actual_u=rng.normal(size=(6, 9)),
                 scatter_u=np.broadcast_to(np.eye(9), (6, 9, 9)).copy())
    ref = dict(amplitude_radii=np.array([.8, 1., 1.5, 2.]),
               cal_log_amplitude=np.array([0., .4, 1., 1.4]),
               cal_amp_scatter=np.broadcast_to(np.eye(9), (4, 9, 9)).copy())
    cal = np.arange(4); query = np.arange(4, 6); groups = np.arange(6)
    stats = dict(u_scale=np.ones(9), u_center=np.zeros(9))
    actual = rng.normal(size=6)
    calls = []

    def score(mean, scatter, stats, target, law, weights, candidates, seed, **kwargs):
        calls.append(dict(mean=mean.copy(), scatter=scatter.copy(), target=target.copy(),
                          log_centers=law['log_centers'].copy(), weights=weights.copy(), seed=seed))
        return np.square(candidates[:, 0]-target[0])[None]

    monkeypatch.setattr(math, 'evaluate_candidate_crps', score)
    baseline = calibration_search(prior, ref, dict(local_bandwidth=.5), cal, groups, stats, actual, 81)
    first = deepcopy(calls); calls.clear()
    changed = deepcopy(prior); changed_actual = actual.copy()
    for key in ('mean_u', 'actual_u', 'scatter_u'):
        changed[key][query] = np.nan
    changed_actual[query] = np.nan
    poisoned = calibration_search(changed, ref, dict(local_bandwidth=.5), cal, groups, stats, changed_actual, 81)
    for a, b in zip(baseline, poisoned):
        np.testing.assert_array_equal(a, b)
    assert len(first) == len(calls) == 4
    for j, (left, right) in enumerate(zip(first, calls)):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        np.testing.assert_array_equal(left['log_centers'], np.log(np.delete(ref['amplitude_radii'], j)))
        np.testing.assert_array_equal(left['mean'], prior['mean_u'][j:j+1])
        np.testing.assert_array_equal(left['target'], actual[j:j+1])
        assert left['weights'].shape == (1, 3)

