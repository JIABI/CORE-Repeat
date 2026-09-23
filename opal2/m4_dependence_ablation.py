"""No-refit, development-only diagonal-scatter ablation of saved CORE laws."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import time

import joblib
import numpy as np

from .conditional_joint_error_experiment import OBSERVABLES, observable_forward
from .empirical_radial import variance_multiplier
from .empirical_radial_experiment import LEVELS, score
from .eu_core_distribution import predict_eu_distribution
from .quantile_direct_data import (_cell_record, _first_well, _roles, _saved,
                                  DATASETS, list_cells)

SAMPLES = 100000
REPORT_NAME = "reports/m4_dependence_ablation_20260922_v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=convert) + "\n")
    temporary.replace(path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def save_npz(path, values):
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def diagonal_scatter(scatter):
    scatter = np.asarray(scatter, dtype=float)
    if scatter.ndim != 3 or scatter.shape[1:] != (9, 9):
        raise ValueError("Expected an n by 9 by 9 coordinate scatter")
    diagonal = np.diagonal(scatter, axis1=-2, axis2=-1)
    if not np.isfinite(scatter).all() or np.any(diagonal <= 0):
        raise ValueError("Invalid scatter diagonal")
    result = np.zeros_like(scatter)
    result[:, np.arange(9), np.arange(9)] = diagonal
    return result


def select(ids, expected, probability, k, lam):
    chosen = np.zeros(len(ids), dtype=bool)
    chosen[np.lexsort((np.asarray(ids).astype(str),
                      -(np.asarray(expected) - lam * np.asarray(probability))))[:k]] = True
    return chosen


@lru_cache(maxsize=1)
def _lincs_legacy(project):
    return read_json(Path(project) / "runs/lincs_empirical_radial_20260916_v1/summary.json")


def load_frozen_cell(project, dataset, index):
    """Load saved development state. No fit function and no R4 path are used."""
    project = Path(project).resolve()
    saved, record, folder, metadata_path, preprocessing = _cell_record(str(project), dataset, index)
    rows, _ = _roles(saved, dataset, index, metadata_path)
    q = rows["query"]
    first, chemistry = _first_well(str(project), dataset)
    if dataset == "EU":
        source = project / "runs/eu_core_cc904_20260917_v1" / f"fold_{index}"
        original = read_npz(source / "AMP_EMP_LOCAL.npz")
        state = read_npz(source / "distribution_arrays.npz")
        np.testing.assert_array_equal(state["query_ids"], original["ids"])
        np.testing.assert_array_equal(state["query_mean_u"], original["mean_u"])
        np.testing.assert_array_equal(state["query_scatter_u"], original["scatter_u"])
        law = read_json(source / "distribution_state.json")["law"]
        weights = state["radial_weights"]
        seed = 20260917 + index * 100
        costs = read_json(source / "summary.json")["costs"]
    elif dataset == "LINCS":
        source = folder
        original = read_npz(source / "CORE_ORIGINAL.npz")
        f, h = record["fold"], record["half"]
        prior = next(cell for cell in _lincs_legacy(str(project))["cells"]
                     if cell["fold"] == f and cell["half"] == h)
        radial_path = project / "runs/lincs_empirical_radial_20260916_v1" / f"cell_{f}_{h}_radial.npz"
        state = read_npz(radial_path)
        np.testing.assert_array_equal(state["query_ids"], original["ids"])
        law, weights = prior["laws"]["amplitude_law"], state["local_weights"]
        seed = 20260918 + f * 100 + h
        costs = dict(reference_if_all_new_wells=4 * len(rows["ref"]),
                     reference_if_X_already_available=3 * len(rows["ref"]),
                     reference_if_reusable_new_wells=0,
                     model_train_and_validation_wells=4 * (len(rows["model_train"]) + len(rows["valid"])),
                     distribution_calibration_wells=4 * len(rows["cal"]))
    else:
        source = folder
        original = read_npz(source / "AMP_EMP_LOCAL.npz")
        fitted = joblib.load(source / "CORE_distribution.joblib")
        query = dict(ids=saved["ids"][q], groups=saved["original"]["core"]["groups"][q],
                     X=first[q], chem=chemistry[q], mean_u=original["mean_u"])
        prediction = predict_eu_distribution(fitted, query)
        np.testing.assert_array_equal(prediction["mean_u"], original["mean_u"])
        np.testing.assert_array_equal(prediction["scatter_u"], original["scatter_u"])
        law, weights = prediction["law"], prediction["radial_weights"]
        seed = 20260918 + index * 100
        costs = read_json(source / "summary.json")["costs"]
    np.testing.assert_array_equal(original["ids"], saved["ids"][q])
    global_core = saved["original"]["core"]
    for key in ("actual", "predicted", "p_null", "crps"):
        np.testing.assert_array_equal(original[key], global_core[key][q])
    np.testing.assert_allclose(variance_multiplier(law, weights), original["radial_variance_multiplier"],
                               atol=1e-12, rtol=1e-12)
    stats = read_json(preprocessing)
    raw = original["actual_u"] * np.asarray(stats["u_scale"]) + np.asarray(stats["u_center"])
    actual_check, observable, differences, _ = observable_forward(raw)
    np.testing.assert_allclose(actual_check, original["actual"], atol=1e-10, rtol=1e-10)
    norm2 = np.square(first[q]).sum(axis=1) / first.shape[1]
    absolute = np.log1p(differences * norm2[:, None])
    k = int(global_core["selected_lambda_0.2"][q].sum())
    for lam in (0.2, 0):
        selected = select(original["ids"], original["predicted"], original["p_null"], k, lam)
        np.testing.assert_array_equal(selected, global_core[f"selected_lambda_{lam}"][q])
        original[f"selected_lambda_{lam}"] = selected
    record.update(n_query=len(q), query_k=k, seed=seed, samples=SAMPLES,
                  sources=dict(original=str(source), preprocessing=str(preprocessing),
                               role_metadata=str(metadata_path)), costs=costs)
    return dict(record=record, original=original, stats=stats, law=law, weights=weights,
                obs_actual=observable, absolute_actual=absolute, norm2=norm2,
                groups=global_core["groups"][q], layout=global_core["layout"][q])


def evaluate(cell, scatter, limit=None):
    o = cell["original"]
    take = slice(None, limit)
    return score(o["mean_u"][take], scatter[take], o["actual_u"][take], cell["stats"],
                 o["actual"][take], cell["obs_actual"][take], cell["absolute_actual"][take],
                 cell["norm2"][take], cell["record"]["seed"], law=cell["law"],
                 weights=cell["weights"][take], samples=SAMPLES)


def verify_dataset(project, dataset, output):
    begin = time.monotonic()
    cell = load_frozen_cell(project, dataset, 0)
    replay = evaluate(cell, cell["original"]["scatter_u"], limit=16)
    differences = {}
    for key, values in replay.items():
        reference = cell["original"][key][:16]
        np.testing.assert_allclose(values, reference, rtol=1e-10, atol=1e-10,
                                   err_msg=f"{dataset}: cached full CORE mismatch for {key}")
        differences[key] = float(np.max(np.abs(values - reference)))
    result = dict(dataset=dataset, n_replayed=16, samples=SAMPLES, seed=cell["record"]["seed"],
                  original_chunk_size=16, max_absolute_differences=differences,
                  wall_seconds=time.monotonic() - begin, passed=True,
                  interpretation="Evaluator replay only; no new fitted model or hyperparameter selection")
    write_json(Path(output) / "verification" / f"{dataset}.json", result)
    return result


def run_cell(project, record, output):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    output = Path(output)
    folder = output / "cells" / record["dataset"] / f"cell_{record['cell_index']:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / "complete.json"
    if completed.exists():
        return read_json(completed)
    with threadpool_limits(limits=1):
        start, cpu = time.monotonic(), time.process_time()
        cell = load_frozen_cell(project, record["dataset"], record["cell_index"])
        write_json(folder / "status.json", dict(state="RUNNING", record=cell["record"]))
        original = cell["original"]
        diagonal = diagonal_scatter(original["scatter_u"])
        np.testing.assert_array_equal(np.diagonal(diagonal, axis1=-2, axis2=-1),
                                      np.diagonal(original["scatter_u"], axis1=-2, axis2=-1))
        scoring_begin = time.monotonic()
        ablated = evaluate(cell, diagonal)
        scoring_seconds = time.monotonic() - scoring_begin
        ablated.update(mean_u=original["mean_u"], actual_u=original["actual_u"], scatter_u=diagonal)
        for arm, values in (("CORE_ORIGINAL", original), ("DIAGONAL_SCATTER", ablated)):
            values.update(ids=original["ids"], groups=cell["groups"], layout=cell["layout"],
                          actual=original["actual"], cell=np.full(len(original["ids"]), record["cell_index"]))
            values["brier"] = (values["p_null"] - (original["actual"] <= 0)) ** 2
            values["two_average_crps"] = values["observable_crps"][:, 6:9].mean(axis=1)
            values["two_average_coverage"] = values["observable_coverage"][:, 6:9].mean(axis=1)
            for lam in (0.2, 0):
                values[f"selected_lambda_{lam}"] = select(original["ids"], values["predicted"],
                                                          values["p_null"], record["query_k"], lam)
            save_npz(folder / f"{arm}.npz", values)
        result = dict(state="COMPLETE", record=cell["record"],
                      wall_seconds=time.monotonic() - start, process_cpu_seconds=time.process_time() - cpu,
                      new_diagonal_score_seconds=scoring_seconds, cpu_threads=1, gpu_used=False,
                      fit_seconds=0, baseline_cached=True, covariance_diagonal_preserved=True,
                      observation_names=list(OBSERVABLES), levels=LEVELS.tolist())
        write_json(completed, result)
        write_json(folder / "status.json", result)
        return result
