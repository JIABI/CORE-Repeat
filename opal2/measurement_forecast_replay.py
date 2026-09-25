"""Replay frozen R4 draws for measurement observables without refitting.

The original eligible-query order and eight-object random-number chunks are
preserved, including queries whose later measurements are missing.  All ten
existing scalar observables are scored in one pass; no selection is changed.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np

from .conditional_joint_error_experiment import OBSERVABLES
from .empirical_radial import draw_radial
from .objective_analysis import fair_crps
from .reference_information_diagnostic import gamma_forward


LEVELS = np.array([.50, .80, .90, .95, .99])
TWO_AVERAGE_INDICES = np.array([6, 7, 8])
PAIRS = ((0, 1), (0, 2), (1, 2))


def _read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _log_norm2(values):
    """log(1 + squared norm), without squaring extreme factor magnitudes."""
    with np.errstate(divide="ignore"):
        log_absolute = np.log(np.abs(values))
    log_norm2 = np.logaddexp.reduce(2 * log_absolute, axis=-1)
    return np.logaddexp(0., log_norm2)


def observable_from_raw(raw):
    """The existing ten factor-forward log-norm observables, without decoding.

    Direct factor forwarding also handles squared-diagonal underflow, which
    is not a reason to discard a finite draw.  No clipping or resampling occurs.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[-1] != 9 or not np.isfinite(raw).all():
        raise ValueError("Expected finite nine-dimensional geometry")
    future = np.zeros((*raw.shape[:-1], 3, 4), dtype=np.float64)
    future[..., :, 0] = raw[..., :3]
    with np.errstate(over="raise"):
        future[..., 0, 1] = np.exp(raw[..., 3])
        future[..., 1, 1] = raw[..., 4]
        future[..., 1, 2] = np.exp(raw[..., 5])
        future[..., 2, 1] = raw[..., 6]
        future[..., 2, 2] = raw[..., 7]
        future[..., 2, 3] = np.exp(raw[..., 8])
    output = np.empty((*raw.shape[:-1], len(OBSERVABLES)), dtype=np.float64)
    output[..., :3] = _log_norm2(future)
    for j, (a, b) in enumerate(PAIRS):
        output[..., 3 + j] = _log_norm2(future[..., a, :] - future[..., b, :])
        output[..., 6 + j] = _log_norm2((future[..., a, :] + future[..., b, :]) / 2)
    output[..., 9] = _log_norm2(future.mean(axis=-2))
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite observable; no draw was dropped")
    return output


def observables_from_profiles(future, initial_norm2):
    """Observed scalar targets directly from complete future well profiles."""
    future = np.asarray(future, dtype=np.float64)
    norm2 = np.asarray(initial_norm2, dtype=np.float64)
    if (future.ndim != 3 or future.shape[1] != 3 or norm2.shape != (len(future),)
            or not np.isfinite(future).all() or not np.isfinite(norm2).all()
            or np.any(norm2 <= 0)):
        raise ValueError("Finite three-role profiles and positive initial norms required")
    relative = future / np.sqrt(norm2)[:, None, None]
    output = np.empty((len(future), len(OBSERVABLES)), dtype=float)
    output[:, :3] = _log_norm2(relative)
    for j, (a, b) in enumerate(PAIRS):
        output[:, 3 + j] = _log_norm2(relative[:, a] - relative[:, b])
        output[:, 6 + j] = _log_norm2((relative[:, a] + relative[:, b]) / 2)
    output[:, 9] = _log_norm2(relative.mean(axis=1))
    return output


def replay_frozen_observables(
        prediction_dir, outcome_file, output_path, *, samples=100000,
        seed=20260921, chunk_size=8, verify_gamma=True, progress=True,
        strict_frozen_config=True):
    """Write new per-query observable summaries and return a compact audit.

    ``outcome_file`` is the released FMP ``outcomes.npz`` containing all
    qualified IDs and three future profiles.  Missing outcomes get NaN scores,
    but predictions and the random stream retain every eligible initial query.
    ``strict_frozen_config=False`` is reserved for small synthetic unit tests.
    Output is a new NPZ plus a same-stem JSON; existing R4 files are read only.
    """
    started = time.monotonic()
    root = Path(prediction_dir).resolve()
    outcome_file = Path(outcome_file).resolve()
    output = Path(output_path).resolve()
    audit_path = output.with_suffix(".json")
    if output.suffix != ".npz":
        raise ValueError("Observable output must be a new .npz file")
    if output.is_relative_to(root.parent):
        raise ValueError("New outputs must be outside the frozen R4 run")
    if output.exists() or audit_path.exists():
        raise FileExistsError("Preserve existing replay outputs")
    if samples < 2 or chunk_size < 1 or chunk_size > 8:
        raise ValueError("At least two samples and one to eight objects per chunk required")
    if strict_frozen_config and (samples, seed, chunk_size) != (100000, 20260921, 8):
        raise ValueError("The final EU replay freezes 100000 samples, seed and chunk size")
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("state") != "COMPLETE" or manifest["samples"] != samples
            or manifest["primary_seed"] != seed or manifest["chunk_size"] != chunk_size):
        raise ValueError("Replay configuration differs from frozen prediction manifest")
    original = _read_npz(root / "CORE_ORIGINAL.npz")
    meta = _read_npz(root / "query_metadata.npz")
    ids = np.asarray(original["ids"], str)
    np.testing.assert_array_equal(ids, np.asarray(manifest["query_ids"], str))
    np.testing.assert_array_equal(ids, meta["ids"])
    n = len(ids)
    if len(set(ids)) != n or (strict_frozen_config and n != 1527):
        raise ValueError("Expected all original eligible-query identities in frozen order")
    mean, scatter = original["mean_u"], original["scatter_u"]
    if mean.shape != (n, 9) or scatter.shape != (n, 9, 9):
        raise ValueError("Frozen mean/scatter arrays have unexpected dimensions")
    stats = json.loads((root / "preprocessing.json").read_text())
    scale, center = np.asarray(stats["u_scale"]), np.asarray(stats["u_center"])
    if scale.shape != (9,) or center.shape != (9,):
        raise ValueError("Frozen coordinate transform must have nine dimensions")
    distribution = joblib.load(root / "query_distribution.joblib")
    if "ids" in distribution:
        np.testing.assert_array_equal(ids, distribution["ids"])
    law, weights = distribution["law"], distribution["radial_weights"]
    if len(weights) != n:
        raise ValueError("Radial weights do not cover the full original query order")

    outcomes = _read_npz(outcome_file)
    outcome_ids = np.asarray(outcomes["ids"], str)
    if len(set(outcome_ids)) != len(outcome_ids):
        raise ValueError("Future-outcome identities must be unique")
    lookup = {oid: i for i, oid in enumerate(outcome_ids)}
    if any(oid not in lookup for oid in ids):
        raise ValueError("A frozen eligible query is absent from the outcome ledger")
    rows = np.array([lookup[oid] for oid in ids], dtype=int)
    future = np.asarray(outcomes["future"][rows], dtype=float)
    if future.ndim != 3 or future.shape[:2] != (n, 3):
        raise ValueError("Expected three future roles per frozen query")
    if "role_order" in outcomes:
        expected_roles = np.array(["Z1", "Z2", "V"])
        np.testing.assert_array_equal(np.asarray(outcomes["role_order"], str), expected_roles)
    for key in ("groups", "layout"):
        if key in outcomes:
            np.testing.assert_array_equal(meta[key], outcomes[key][rows])
    observed = np.isfinite(future).all(axis=(1, 2))
    if strict_frozen_config and observed.sum() != 1520:
        raise ValueError("Expected the fixed 1520 complete confirmation objects")
    initial_norm2 = np.asarray(meta["norm2_per_feature"], float) * future.shape[2]
    actual = np.full((n, len(OBSERVABLES)), np.nan)
    actual[observed] = observables_from_profiles(future[observed], initial_norm2[observed])
    del future, outcomes

    result = dict(ids=ids, groups=meta["groups"], layout=meta["layout"],
                  observed=observed, actual=actual, levels=LEVELS.copy(),
                  observable_names=np.array(OBSERVABLES),
                  two_average_indices=TWO_AVERAGE_INDICES.copy())
    for key in ("mean", "mean_mc_se", "crps"):
        result[key] = np.full((n, len(OBSERVABLES)), np.nan)
    for key in ("lower", "upper", "width", "coverage"):
        result[key] = np.full((n, len(OBSERVABLES), len(LEVELS)), np.nan)
    gamma_cache = None
    if verify_gamma:
        gamma_cache = np.load(root / "CORE_ORIGINAL_gamma_samples.npy", mmap_mode="r",
                              allow_pickle=False)
        if gamma_cache.shape != (n, samples):
            raise ValueError("Frozen Gamma cache dimensions differ from replay")
    max_gamma_difference = 0.
    normal_rng, radius_rng = np.random.default_rng(seed), np.random.default_rng(seed + 47000)
    probabilities = np.r_[(1 - LEVELS) / 2, (1 + LEVELS) / 2]
    for begin in range(0, n, chunk_size):
        end = min(begin + chunk_size, n)
        normal = normal_rng.normal(size=(samples, end - begin, 9))
        eps = draw_radial(law, weights[begin:end], scatter[begin:end], normal,
                          radius_rng.random((samples, end - begin)),
                          radius_rng.random((samples, end - begin)))
        u = mean[None, begin:end] + eps
        raw = u * scale + center
        del normal, eps, u
        if gamma_cache is not None:
            gamma = gamma_forward(raw)
            cached = gamma_cache[begin:end].T
            difference = float(np.max(np.abs(gamma - cached)))
            max_gamma_difference = max(max_gamma_difference, difference)
            np.testing.assert_allclose(gamma, cached, rtol=1e-12, atol=1e-12,
                                       err_msg=f"Frozen Gamma stream differs at rows {begin}:{end}")
            del gamma, cached
        draws = observable_from_raw(raw)
        del raw
        valid_chunk = observed[begin:end]
        valid_rows = np.flatnonzero(valid_chunk) + begin
        for j in range(len(OBSERVABLES)):
            values = draws[..., j]
            result["mean"][begin:end, j] = values.mean(axis=0)
            result["mean_mc_se"][begin:end, j] = values.std(axis=0, ddof=1) / np.sqrt(samples)
            quantiles = np.quantile(values, probabilities, axis=0)
            lower, upper = quantiles[:len(LEVELS)].T, quantiles[len(LEVELS):].T
            result["lower"][begin:end, j] = lower
            result["upper"][begin:end, j] = upper
            result["width"][begin:end, j] = upper - lower
            if len(valid_rows):
                truth = actual[valid_rows, j]
                result["crps"][valid_rows, j] = fair_crps(values[:, valid_chunk], truth)
                result["coverage"][valid_rows, j] = (
                    (truth[:, None] >= lower[valid_chunk]) &
                    (truth[:, None] <= upper[valid_chunk]))
        del draws, values
        if progress:
            print(f"frozen observable replay: {end}/{n}; Gamma max error={max_gamma_difference:.3g}",
                  flush=True)
    result["two_average_crps"] = result["crps"][:, TWO_AVERAGE_INDICES].mean(axis=1)
    audit = dict(status="COMPLETE", prediction_dir=str(root), outcome_file=str(outcome_file),
                 output_path=str(output), n_predictions=n, n_complete=int(observed.sum()),
                 samples=samples, seed=seed, radial_seed=seed + 47000, chunk_size=chunk_size,
                 gamma_parity_checked=bool(verify_gamma),
                 gamma_max_absolute_difference=max_gamma_difference if verify_gamma else None,
                 observable_names=list(OBSERVABLES), two_average_indices=TWO_AVERAGE_INDICES.tolist(),
                 levels=LEVELS.tolist(), fitting=False, selections_changed=False,
                 full_eligible_order_preserved=True, strict_frozen_config=bool(strict_frozen_config),
                 definition="log(1 + squared profile norm / squared initial-profile norm)",
                 crps="fair Monte Carlo CRPS; ensemble pair denominator S*(S-1)",
                 missing_outcomes="predictions retained; observed targets and scores remain NaN",
                 full_joint_draws_saved=False, elapsed_seconds=time.monotonic() - started)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **result)
    with audit_path.open("x") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return audit
