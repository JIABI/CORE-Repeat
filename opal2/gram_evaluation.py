"""One outcome-preserving evaluator for joint spectrum and direct Gram models."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .biology_kernel_evaluation import write_json
from .baseline_policy import evaluate_predictions, ACTIONS
from .gram_geometry import gram_gains, gram_observables, OBSERVABLE_NAMES
from .objective_analysis import fair_crps


FREE_ROWS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3])
FREE_COLS = np.array([1, 2, 3, 1, 2, 3, 2, 3, 3])


def free_entries(grams):
    array = np.asarray(grams, dtype=np.float64)
    if array.shape[-2:] != (4, 4) or not np.isfinite(array).all():
        raise ValueError("Finite four-well Gram matrices required")
    return array[..., FREE_ROWS, FREE_COLS]


def fit_score_scale(train_grams):
    """Fit a single common score metric; no held-out information is used."""
    values = free_entries(train_grams)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("At least two training Gram matrices required")
    scale = np.std(values, axis=0, ddof=0)
    return np.where(scale >= 1e-6, scale, 1.0)


def paired_energy_score(samples, observed):
    """Unbiased linear-cost Monte Carlo energy score using disjoint IID pairs.

    Correlations among coordinates/objects within a draw are retained. Different
    Monte Carlo draws, not different objects, provide the independent pairs.
    """
    samples, observed = np.asarray(samples), np.asarray(observed)
    if samples.ndim != 3 or samples.shape[1:] != observed.shape or len(samples) < 2:
        raise ValueError("Energy score expects [S,N,K] and [N,K]")
    if not np.isfinite(samples).all() or not np.isfinite(observed).all():
        raise ValueError("Energy score does not omit nonfinite objects")
    count = len(samples) // 2
    first = np.linalg.norm(samples - observed[None], axis=-1).mean(0)
    second = np.linalg.norm(samples[:count] - samples[count:2*count], axis=-1).mean(0)
    return first - 0.5 * second


def _arrays(grams):
    tensor = torch.as_tensor(np.asarray(grams), dtype=torch.float64)
    return gram_observables(tensor).numpy(), gram_gains(tensor).numpy()


def predictive_summary(draws, observed, names):
    draws, observed = np.asarray(draws), np.asarray(observed)
    mean = draws.mean(0)
    crps = fair_crps(draws, observed)
    rows = []
    for j, name in enumerate(names):
        error = mean[:, j] - observed[:, j]
        squared = error**2
        rows.append(dict(name=name, observed_mean=observed[:, j].mean(),
            predictive_mean=mean[:, j].mean(), mean_bias=error.mean(),
            mean_prediction_mse=squared.mean(), median_squared_error=np.median(squared),
            max_squared_error_fraction=squared.max()/squared.sum() if squared.sum() else None,
            observed_sd=observed[:, j].std(), predicted_mean_sd=mean[:, j].std(),
            pooled_predictive_sd=draws[:, :, j].std(), crps=crps[:, j].mean(), intervals=[]))
    for level in (.5, .8, .9, .95):
        low, high = np.quantile(draws, [(1-level)/2, (1+level)/2], axis=0)
        covered = (observed >= low) & (observed <= high)
        for j, row in enumerate(rows):
            row["intervals"].append(dict(level=level, coverage=covered[:, j].mean(),
                mean_width=(high[:, j]-low[:, j]).mean()))
    return rows, crps


def score_grams(samples, actual_grams, ids, *, train_actual_gains, score_scale,
                seed=20260914, n_bootstrap=2000, n_random=2000):
    """Score the identical marginal per-object law, never Gamma of mean Gram."""
    samples = np.asarray(samples, dtype=np.float64)
    actual_grams = np.asarray(actual_grams, dtype=np.float64)
    if samples.ndim != 4 or samples.shape[1:] != actual_grams.shape or len(samples) < 2:
        raise ValueError("Expected [S,N,4,4] samples and [N,4,4] targets")
    if not np.allclose(samples[..., 0, 0], 1., rtol=1e-10, atol=1e-10) or not np.allclose(actual_grams[..., 0, 0], 1., rtol=1e-10, atol=1e-10):
        raise ValueError("All energy scores require the same X-normalized G00=1 geometry")
    if len(ids) != len(actual_grams) or len(set(map(str, ids))) != len(ids):
        raise ValueError("Each scored object must have one unique ID")
    scale = np.asarray(score_scale, dtype=np.float64)
    if scale.shape != (9,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Nine positive TRAIN-fitted score scales required")
    obs, gains = _arrays(samples)
    actual_obs, actual = _arrays(actual_grams)
    mean, pnull = gains.mean(0), (gains <= 0).mean(0)
    geometry, geometry_crps = predictive_summary(obs, actual_obs, OBSERVABLE_NAMES)
    utility, utility_crps = predictive_summary(gains, actual, ACTIONS)
    energy = paired_energy_score(free_entries(samples)/scale, free_entries(actual_grams)/scale)
    policy = evaluate_predictions(mean, pnull, actual, ids, fractions=(.05, .10, .25),
        train_actual=train_actual_gains, seed=seed, n_bootstrap=n_bootstrap, n_random=n_random)
    calibration = []
    for j, action in enumerate(ACTIONS):
        bins = []
        for k in range(5):
            selected = (pnull[:, j] >= k/5) & ((pnull[:, j] < (k+1)/5) if k < 4 else (pnull[:, j] <= 1))
            bins.append(dict(lower=k/5, upper=(k+1)/5, n=int(selected.sum()),
                predicted=pnull[selected, j].mean() if selected.any() else None,
                observed=(actual[selected, j] <= 0).mean() if selected.any() else None))
        calibration.append(dict(action=action, bins=bins))
    report = dict(n=len(ids), samples=len(samples), actions=list(ACTIONS),
        geometry=geometry, utility=utility, action_metrics=policy["action_metrics"],
        joint_geometry_energy_score=float(energy.mean()), null_calibration=calibration,
        mean_utility_mc_se=(gains.std(0, ddof=1)/np.sqrt(len(gains))).mean(0),
        policy=policy, formal_certificate=False, historical_dev=True,
        endpoint_changed=False, independent_new_holdout=False,
        score_scope="per-object predictive laws in the original utility, conditional on the fixed shared batches",
        geometry_scope="norm/angle distributions; not coordinate intervals or identified variance components",
        density_comparison="no comparison of spectrum NLL with geometry-coordinate NLL")
    traces = dict(actual=actual, predicted=mean, p_null=pnull, utility_crps=utility_crps,
        geometry_crps=geometry_crps, geometry_energy=energy, actual_observables=actual_obs,
        predicted_observables=obs.mean(0), utility_samples=gains,
        actual_grams=actual_grams, score_scale=scale)
    return report, traces


def evaluate_and_save(path, samples, actual_grams, ids, *, metadata, **kwargs):
    path = Path(path)
    if (path / "metrics.json").exists():
        raise FileExistsError("Completed Gram evaluation is not overwritten")
    path.mkdir(parents=True, exist_ok=True)
    report, traces = score_grams(samples, actual_grams, ids, **kwargs)
    report["model"] = metadata
    np.savez_compressed(path / "predictions.npz", ids=np.asarray(ids, dtype=str), **traces)
    np.savez_compressed(path / "joint_grams.npz", ids=np.asarray(ids, dtype=str),
                        grams=np.asarray(samples, dtype=np.float64), actual_grams=actual_grams)
    write_json(path / "metrics.json", report)
    return report


def paired_score_comparison(left, right, *, seed=20260914, n_bootstrap=2000):
    """Left minus right: negative score differences favor left, not a certificate."""
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if not np.array_equal(a["ids"], b["ids"]):
            raise ValueError("Pairwise comparisons require identical object order")
        for key in ("actual", "actual_grams", "score_scale"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError("Pairwise comparison has mismatched endpoint or score scale: "+key)
        differences = {"geometry_energy": a["geometry_energy"]-b["geometry_energy"],
            **{f"gamma_crps_{name}": a["utility_crps"][:, j]-b["utility_crps"][:, j]
               for j, name in enumerate(ACTIONS)},
            **{f"null_brier_{name}": (a["p_null"][:, j]-(a["actual"][:, j] <= 0))**2
               -(b["p_null"][:, j]-(b["actual"][:, j] <= 0))**2 for j, name in enumerate(ACTIONS)}}
    rng = np.random.default_rng(seed)
    result = {}
    for name, values in differences.items():
        draws = rng.integers(len(values), size=(n_bootstrap, len(values)))
        result[name] = dict(mean=float(values.mean()),
            interval95=np.quantile(values[draws].mean(1), [.025, .975]).tolist())
    return dict(differences=result, resampling_unit="compound",
        scope="conditional on these fitted predictions and shared batches; no selection uncertainty",
        formal_certificate=False)
