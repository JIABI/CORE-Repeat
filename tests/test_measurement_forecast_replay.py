"""Small synthetic checks of the exact frozen stream and observable interface."""
import json

import joblib
import numpy as np
import pytest

from opal2.conditional_joint_error_experiment import observable_forward
from opal2.empirical_radial import fit_radial
from opal2.measurement_forecast_replay import (
    observable_from_raw, observables_from_profiles, replay_frozen_observables,
)
from opal2.objective_analysis import fair_crps
from opal2.r4_final_model import integrate


def test_observables_match_existing_factor_forward_and_handle_extreme_norms():
    raw = np.random.default_rng(7).normal(size=(40, 4, 9))
    np.testing.assert_allclose(observable_from_raw(raw), observable_forward(raw)[1],
                               atol=2e-15, rtol=2e-15)
    raw = np.zeros((2, 9))
    raw[0, [3, 5, 8]] = -400.  # squared diagonal underflows in a strict decoder
    raw[1, [3, 5, 8]] = 400.   # squared norm overflows without the log-domain sum
    out = observable_from_raw(raw)
    assert np.isfinite(out).all() and (out >= 0).all()
    assert out[1, 0] == pytest.approx(800.)


def test_targets_from_real_profiles_have_correct_relative_scale():
    y = np.random.default_rng(14).normal(size=(5, 3, 11))
    norm2 = np.arange(1., 6.)
    got = observables_from_profiles(y, norm2)
    for j, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        expected = np.log1p(np.square((y[:, a] + y[:, b]) / 2).sum(1) / norm2)
        np.testing.assert_allclose(got[:, 6 + j], expected, atol=1e-15)


def _synthetic_files(tmp_path):
    root = tmp_path / "frozen" / "predictions"
    root.mkdir(parents=True)
    n, samples, seed, chunk = 5, 120, 91, 2
    ids = np.array([f"q{i}" for i in range(n)])
    stats = dict(u_scale=np.ones(9).tolist(), u_center=np.zeros(9).tolist())
    mean = np.random.default_rng(8).normal(size=(n, 9)) * .05
    scatter = np.tile(np.eye(9)[None] * .1, (n, 1, 1))
    law = fit_radial(np.array([1., 2., 3.]))
    weights = np.full((n, 3), 1 / 3)
    original = integrate(mean, scatter, stats, seed=seed, samples=samples,
                         chunk_size=chunk, law=law, weights=weights,
                         gamma_path=root / "CORE_ORIGINAL_gamma_samples.npy")
    np.savez_compressed(root / "CORE_ORIGINAL.npz", ids=ids, mean_u=mean,
                        scatter_u=scatter, **original)
    groups, layout = ids.copy(), np.array(["A", "A", "B", "B", "C"])
    np.savez_compressed(root / "query_metadata.npz", ids=ids, groups=groups,
                        layout=layout, norm2_per_feature=np.ones(n))
    (root / "preprocessing.json").write_text(json.dumps(stats))
    (root / "manifest.json").write_text(json.dumps(dict(state="COMPLETE", query_ids=ids.tolist(),
        primary_seed=seed, samples=samples, chunk_size=chunk)))
    joblib.dump(dict(ids=ids, law=law, radial_weights=weights), root / "query_distribution.joblib")
    future = np.random.default_rng(19).normal(size=(n, 3, 12))
    future[1] = np.nan  # an early missing row must still consume its frozen draws
    future[4, 2] = np.nan
    # Deliberately different outcome order proves ID alignment, not row matching.
    order = np.array([4, 1, 3, 0, 2])
    outcome_path = tmp_path / "future.npz"
    np.savez_compressed(outcome_path, ids=ids[order], groups=groups[order],
                        layout=layout[order], future=future[order],
                        role_order=np.array(["Z1", "Z2", "V"]))
    return root, outcome_path, samples, seed, chunk


def test_replay_keeps_missing_queries_and_matches_every_cached_gamma(tmp_path):
    root, outcomes, samples, seed, chunk = _synthetic_files(tmp_path)
    out = tmp_path / "new" / "observables.npz"
    audit = replay_frozen_observables(root, outcomes, out, samples=samples, seed=seed,
        chunk_size=chunk, strict_frozen_config=False, progress=False)
    assert audit["n_predictions"] == 5 and audit["n_complete"] == 3
    assert audit["gamma_max_absolute_difference"] == 0
    with np.load(out, allow_pickle=False) as z:
        assert z["observed"].tolist() == [True, False, True, True, False]
        assert z["crps"].shape == (5, 10)
        assert z["coverage"].shape == (5, 10, 5)
        assert np.isfinite(z["mean"]).all()
        assert np.isfinite(z["crps"][[0, 2, 3]]).all()
        assert np.isnan(z["actual"][[1, 4]]).all()
        assert np.isnan(z["crps"][[1, 4]]).all()
        np.testing.assert_array_equal(z["two_average_crps"], z["crps"][:, 6:9].mean(1))
        assert np.all(z["width"] >= 0)
    with pytest.raises(FileExistsError):
        replay_frozen_observables(root, outcomes, out, samples=samples, seed=seed,
            chunk_size=chunk, strict_frozen_config=False, progress=False)
    with pytest.raises(ValueError, match="outside the frozen"):
        replay_frozen_observables(root, outcomes, root / "new.npz", samples=samples,
            seed=seed, chunk_size=chunk, strict_frozen_config=False, progress=False)


def test_changed_gamma_cache_is_rejected(tmp_path):
    root, outcomes, samples, seed, chunk = _synthetic_files(tmp_path)
    path = root / "CORE_ORIGINAL_gamma_samples.npy"
    z = np.load(path, mmap_mode="r+")
    z[3, 7] += .001
    z.flush()
    with pytest.raises(AssertionError, match="Frozen Gamma stream"):
        replay_frozen_observables(root, outcomes, tmp_path / "new.npz", samples=samples,
            seed=seed, chunk_size=chunk, strict_frozen_config=False, progress=False)
