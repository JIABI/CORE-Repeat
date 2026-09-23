#!/usr/bin/env python3
"""Full, resumable C3 conditional-quantile HistGB development experiment."""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
import warnings

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from opal2.quantile_direct_data import list_cells, load_cell
from opal2.quantile_distribution import GRID, make_quantile_law, fit_quantile_offsets

RUN_NAME = "c3_quantile_histgb_20260922_v1"
DEFAULT_RUN = PROJECT / "runs" / RUN_NAME
PROTOCOL = PROJECT / "protocols/C3_conditional_quantile_HistGB_20260922.md"
PARAMETERS = ((7, 20), (15, 10), (31, 10))
COVERAGES = (.5, .8, .9, .95, .99)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def memory_high_water_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024**2 if sys.platform == "darwin" else 1024))


def check_disk(folder):
    free = shutil.disk_usage(folder).free
    if free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free: stopped before writing more checkpoints; no files removed")
    return free


def configuration():
    return dict(
        experiment=RUN_NAME, sklearn_version=sklearn.__version__,
        class_name="sklearn.ensemble.HistGradientBoostingRegressor",
        loss="quantile", levels=GRID.tolist(),
        candidates=[dict(max_leaf_nodes=a, min_samples_leaf=b) for a, b in PARAMETERS],
        max_iter=200, learning_rate=.05, l2_regularization=1., max_bins=255,
        max_features=1., max_depth=None, early_stopping=False,
        selection="VALIDATION exact reconstructed-law CRPS, one setting for all quantiles",
        tie_atol=1e-12, tie_rtol=1e-12, refit_after_validation=False,
        fitting_access="original TRAIN plus REF_FIT; original full ACCESS_MATCHED inputs",
        primary="raw conditional-quantile law", secondary="DIST_CAL per-quantile offsets",
        bounds=[-1.02, .98], interval_levels=list(COVERAGES),
        ranking_lambda=.2, secondary_ranking_lambda=0., bootstrap_repetitions=2000,
        fit_threads_per_worker=1, protocol=str(PROTOCOL),
        r4_used=False, monte_carlo_samples=0,
    )


def ordered_cells(cells):
    """Start one full EU and Rx cell, then interleave remaining real cells."""
    first = [c for key in (("EU", 0), ("RxRx3", 0)) for c in cells
             if (c["dataset"], c["cell_index"]) == key]
    remaining = [c for c in cells if (c["dataset"], c["cell_index"])
                 not in {("EU", 0), ("RxRx3", 0)}]
    order = {"JUMP": 0, "LINCS": 1, "EU": 2, "RxRx3": 3}
    return first + sorted(remaining, key=lambda c: (c["cell_index"], order[c["dataset"]]))


def prepare(run_root):
    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    cfg = configuration()
    if sklearn.__version__ != "1.9.1":
        raise RuntimeError("Protocol specifies installed scikit-learn 1.9.1; do not silently change implementation")
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    path = run_root / "configuration.json"
    if path.exists() and read_json(path) != cfg:
        raise RuntimeError("Existing C3 configuration differs; refusing to overwrite or mix runs")
    cells = list_cells(PROJECT)
    assert len(cells) == 60
    if sum(c["n_query"] for c in cells) != 13141:
        raise ValueError("Unexpected full development cohort size")
    check_disk(run_root)
    write_json(path, cfg)
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["cells"] != cells:
            raise RuntimeError("Saved cell identities/counts changed")
    else:
        manifest = dict(created_at=now(), cells=cells, n_cells=60,
                        quantile_fits_per_cell=57, n_quantile_fits=3420,
                        protocol_text=PROTOCOL.read_text(), configuration=cfg)
        write_json(manifest_path, manifest)
    return manifest


def cell_folder(run_root, cell):
    return Path(run_root) / cell["dataset"] / f"cell_{cell['cell_index']}"


def cell_status(folder, state, **fields):
    write_json(folder / "status.json", dict(state=state, updated_at=now(), pid=os.getpid(), **fields))


def fit_one_cell(project, run_root, cell):
    """New fitting uses TRAIN/VALIDATION/CAL only; query labels enter scoring last."""
    folder = cell_folder(run_root, cell)
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / "complete.json"
    if completed.exists():
        if not (folder / "query_predictions.npz").exists():
            raise RuntimeError("Completion marker has no prediction artifact")
        return dict(dataset=cell["dataset"], cell_index=cell["cell_index"], reused=True)
    started = time.perf_counter()
    cpu_started = time.process_time()
    cell_status(folder, "loading", label=cell["label"], completed_quantile_fits=0)
    try:
        with threadpool_limits(limits=1):
            data = load_cell(project, cell["dataset"], cell["cell_index"])
            lineage = dict(cell=cell, seed=data["seed"], source_paths=data["source_paths"],
                           full_input_dimension=data["full_input_dimension"],
                           ids={role: data["ids_" + role].astype(str).tolist()
                                for role in ("train", "model_train", "ref", "valid", "cal", "query")})
            lineage_path = folder / "cell_manifest.json"
            if lineage_path.exists() and read_json(lineage_path) != lineage:
                raise RuntimeError("Cell fitting data/order changed since checkpointing")
            write_json(lineage_path, lineage)
            all_metadata = []
            candidates = []
            fitted_this_session = 0
            reused = 0
            candidate_valid = []
            for candidate_index, (leaves, minimum) in enumerate(PARAMETERS):
                candidate_folder = folder / f"candidate_{candidate_index}"
                candidate_folder.mkdir(exist_ok=True)
                valid_columns = []
                for quantile_index, alpha in enumerate(GRID):
                    checkpoint = candidate_folder / f"quantile_{quantile_index:02d}.joblib"
                    serial = candidate_index * len(GRID) + quantile_index
                    cell_status(folder, "fitting", label=cell["label"],
                                candidate_index=candidate_index, quantile_index=quantile_index,
                                quantile=float(alpha), completed_quantile_fits=serial,
                                total_quantile_fits=57, elapsed_seconds=time.perf_counter()-started,
                                fitted_this_session=fitted_this_session, reused_quantile_fits=reused)
                    if checkpoint.exists():
                        bundle = joblib.load(checkpoint)
                        reused += 1
                    else:
                        check_disk(folder)
                        model = HistGradientBoostingRegressor(
                            loss="quantile", quantile=float(alpha),
                            max_leaf_nodes=leaves, min_samples_leaf=minimum,
                            max_iter=200, learning_rate=.05, l2_regularization=1.,
                            max_bins=255, max_features=1., max_depth=None,
                            early_stopping=False, random_state=data["seed"],
                            categorical_features=None,
                        )
                        fit_start = time.perf_counter()
                        fit_cpu = time.process_time()
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model.fit(data["x_train"], data["y_train"])
                        fit_seconds = time.perf_counter() - fit_start
                        fit_cpu_seconds = time.process_time() - fit_cpu
                        predict_start = time.perf_counter()
                        valid_prediction = model.predict(data["x_valid"])
                        metadata = dict(
                            candidate_index=candidate_index, quantile_index=quantile_index,
                            quantile=float(alpha), seed=data["seed"], n_iter=int(model.n_iter_),
                            fit_wall_seconds=fit_seconds, fit_cpu_seconds=fit_cpu_seconds,
                            validation_prediction_seconds=time.perf_counter()-predict_start,
                            warnings=[str(w.message) for w in caught],
                            parameters=model.get_params(),
                        )
                        if model.n_iter_ != 200:
                            raise RuntimeError("A quantile model did not complete its full 200-iteration budget")
                        bundle = dict(model=model, valid_prediction=valid_prediction, metadata=metadata)
                        temporary = checkpoint.with_name(checkpoint.name + f".tmp-{os.getpid()}")
                        joblib.dump(bundle, temporary, compress=3)
                        temporary.replace(checkpoint)
                        write_json(checkpoint.with_suffix(".json"), metadata)
                        fitted_this_session += 1
                    meta = bundle["metadata"]
                    if (meta["candidate_index"] != candidate_index
                            or meta["quantile_index"] != quantile_index
                            or meta["seed"] != data["seed"] or meta["n_iter"] != 200):
                        raise RuntimeError("Quantile checkpoint does not match this cell/configuration")
                    valid_columns.append(bundle["valid_prediction"])
                    all_metadata.append(meta)
                    del bundle
                    if "model" in locals():
                        del model
                valid_raw = np.column_stack(valid_columns)
                valid_law = make_quantile_law(valid_raw)
                loss = float(valid_law.crps(data["y_valid"]).mean())
                if not np.isfinite(loss):
                    raise ValueError("Nonfinite VALIDATION CRPS")
                candidates.append(dict(index=candidate_index, max_leaf_nodes=leaves,
                                       min_samples_leaf=minimum, validation_crps=loss))
                candidate_valid.append(valid_raw)
                write_json(folder / "validation_candidates.json", candidates)
            selected = 0
            for index in range(1, len(candidates)):
                old, new = candidates[selected]["validation_crps"], candidates[index]["validation_crps"]
                if new < old and not np.isclose(new, old, atol=1e-12, rtol=1e-12):
                    selected = index
            cell_status(folder, "predicting", label=cell["label"], completed_quantile_fits=57,
                        selected_candidate=selected, elapsed_seconds=time.perf_counter()-started)
            prediction_start = time.perf_counter()
            cal_columns, query_columns = [], []
            for qi in range(len(GRID)):
                bundle = joblib.load(folder / f"candidate_{selected}/quantile_{qi:02d}.joblib")
                cal_columns.append(bundle["model"].predict(data["x_cal"]))
                query_columns.append(bundle["model"].predict(data["x_query"]))
                del bundle
            cal_raw = np.column_stack(cal_columns)
            query_raw = np.column_stack(query_columns)
            prediction_seconds = time.perf_counter() - prediction_start
            calibration_start = time.perf_counter()
            offsets = fit_quantile_offsets(cal_raw, data["y_cal"])
            calibration_seconds = time.perf_counter()-calibration_start
            integration_start = time.perf_counter()
            laws = dict(raw=make_quantile_law(query_raw),
                        cal=make_quantile_law(query_raw + offsets))
            # Query outcomes are used only below, after all fitting/selection.
            arrays = dict(
                ids=data["ids_query"].astype(str), groups=data["groups_query"].astype(str),
                layout=data["layout_query"].astype(str), actual=data["y_query"],
                original_k=np.asarray(data["query_k"]), quantile_levels=GRID.copy(),
                nominal_coverage=np.asarray(COVERAGES), quantiles_raw=query_raw,
                calibration_offsets=offsets, selected_candidate=np.asarray(selected),
            )
            for name, law in laws.items():
                intervals = law.interval_metrics(data["y_query"], COVERAGES)
                arrays.update({name+"_expected": law.mean(), name+"_p_null": law.cdf(0.),
                               name+"_crps": law.crps(data["y_query"]),
                               name+"_coverage": intervals["covered"],
                               name+"_width": intervals["width"],
                               name+"_interval_lower": intervals["lower"],
                               name+"_interval_upper": intervals["upper"],
                               name+"_law_quantiles": law.quantiles,
                               name+"_law_probabilities": law.probabilities})
            for name, original in data["originals"].items():
                for field in ("expected", "p_null", "crps", "coverage", "width", "selected"):
                    if field in original:
                        arrays[name+"_"+field] = original[field]
            integration_seconds = time.perf_counter()-integration_start
            write_npz(folder / "query_predictions.npz", **arrays)
            write_npz(folder / "fit_predictions.npz", ids_valid=data["ids_valid"].astype(str),
                      ids_cal=data["ids_cal"].astype(str), y_valid=data["y_valid"], y_cal=data["y_cal"],
                      validation_quantiles=np.asarray(candidate_valid), calibration_quantiles=cal_raw,
                      calibration_offsets=offsets)
            timing = dict(
                fit_wall_seconds=sum(m["fit_wall_seconds"] for m in all_metadata),
                fit_cpu_seconds=sum(m["fit_cpu_seconds"] for m in all_metadata),
                validation_prediction_seconds=sum(m["validation_prediction_seconds"] for m in all_metadata),
                winning_fit_wall_seconds=sum(m["fit_wall_seconds"] for m in all_metadata
                                             if m["candidate_index"] == selected),
                cal_query_prediction_seconds=prediction_seconds,
                calibration_seconds=calibration_seconds, integration_seconds=integration_seconds,
                current_session_wall_seconds=time.perf_counter()-started,
                current_session_cpu_seconds=time.process_time()-cpu_started,
                worker_memory_high_water_mib=memory_high_water_mib(),
                memory_scope="process high-water mark, may include earlier cells; not incremental per-model RAM",
                fitted_this_session=fitted_this_session, reused_quantile_fits=reused,
                historical_core_cost="not rerun and not counted as zero; use existing component cost ledger",
            )
            completion = dict(dataset=cell["dataset"], cell_index=cell["cell_index"],
                              label=cell["label"], completed_at=now(), n_query=len(data["y_query"]),
                              selected_candidate=selected, candidates=candidates,
                              law_metadata=laws["raw"].metadata(), timings=timing,
                              n_quantile_fits=57, n_threads=1, r4_used=False)
            write_json(completed, completion)
            cell_status(folder, "complete", label=cell["label"], completed_quantile_fits=57,
                        selected_candidate=selected, elapsed_seconds=time.perf_counter()-started)
            return dict(dataset=cell["dataset"], cell_index=cell["cell_index"], reused=False,
                        fit_wall_seconds=timing["fit_wall_seconds"],
                        session_wall_seconds=timing["current_session_wall_seconds"])
    except Exception:
        cell_status(folder, "failed", label=cell["label"], error=traceback.format_exc(),
                    elapsed_seconds=time.perf_counter()-started)
        raise


def coordinator(run_root, workers):
    run_root = Path(run_root).resolve()
    manifest = prepare(run_root)
    if workers not in (1, 2):
        raise ValueError("This local protocol uses one or two single-thread workers")
    lock_stream = (run_root / "coordinator.lock").open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("A coordinator already owns this run") from exc
    start = now()
    results, failures = [], []
    state = dict(state="running", started_at=start, updated_at=start, coordinator_pid=os.getpid(),
                 workers=workers, cells_total=60, cells_complete=0, cells_failed=0)
    write_json(run_root / "status.json", state)
    print(f"{start} START {RUN_NAME}; {workers} CPU workers; 60 cells, 3420 full quantile fits", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            pending = {executor.submit(fit_one_cell, str(PROJECT), str(run_root), cell): cell
                       for cell in ordered_cells(manifest["cells"])}
            for future in as_completed(pending):
                cell = pending[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"{now()} COMPLETE {cell['label']} ({len(results)}/60) {json.dumps(result)}", flush=True)
                except Exception:
                    failures.append(dict(cell=cell, error=traceback.format_exc()))
                    print(f"{now()} FAILED {cell['label']}\n{failures[-1]['error']}", flush=True)
                state.update(updated_at=now(), cells_complete=len(results), cells_failed=len(failures))
                write_json(run_root / "status.json", state)
                write_json(run_root / "coordinator_results.json", dict(completed=results, failed=failures))
        if failures:
            raise RuntimeError(f"{len(failures)} cells failed; completed cells remain resumable")
        state.update(state="aggregating", updated_at=now())
        write_json(run_root / "status.json", state)
        from opal2.quantile_direct_evaluation import aggregate
        result = aggregate(PROJECT, run_root)
        if not result.get("complete"):
            raise RuntimeError("All fitting cells returned but four-dataset aggregation is incomplete")
        state.update(state="complete", updated_at=now(), completed_at=now(),
                     aggregate_result=str(result))
        write_json(run_root / "status.json", state)
        print(f"{now()} ALL COMPLETE {result}", flush=True)
    except Exception:
        state.update(state="failed", updated_at=now(), error=traceback.format_exc())
        write_json(run_root / "status.json", state)
        print(state["error"], flush=True)
        raise
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()


def status(run_root):
    run_root = Path(run_root).resolve()
    state_path = run_root / "status.json"
    result = dict(run_root=str(run_root), state="not_started")
    if state_path.exists():
        result.update(read_json(state_path))
    counts = {}
    active = []
    times = []
    for path in sorted(run_root.glob("*/cell_*/status.json")):
        record = read_json(path)
        dataset = path.parents[1].name
        counter = counts.setdefault(dataset, dict(complete=0, fitting=0, failed=0, quantile_fits=0))
        counter["quantile_fits"] += record.get("completed_quantile_fits", 0)
        if record["state"] == "complete":
            counter["complete"] += 1
            completion = read_json(path.with_name("complete.json"))
            times.append(dict(dataset=dataset, cell=completion["cell_index"],
                              fit_seconds=completion["timings"]["fit_wall_seconds"]))
        elif record["state"] == "failed":
            counter["failed"] += 1
            active.append(record)
        else:
            counter["fitting"] += 1
            active.append(record)
    result.update(by_dataset=counts, active_cells=active, completed_cell_fit_seconds=times,
                  disk_free_gib=round(shutil.disk_usage(run_root if run_root.exists() else PROJECT).free / 1024**3, 2))
    pid = result.get("coordinator_pid")
    if pid is not None:
        try:
            os.kill(pid, 0)
            result["coordinator_alive"] = True
        except ProcessLookupError:
            result["coordinator_alive"] = False
    return result


def launch(run_root, workers):
    run_root = Path(run_root).resolve()
    prepare(run_root)
    # Lock checking covers a still-running coordinator even if a PID file is old.
    with (run_root / "coordinator.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("C3 is already running; use status")
        fcntl.flock(stream, fcntl.LOCK_UN)
    if (run_root / "status.json").exists():
        previous = read_json(run_root / "status.json")
        if previous.get("state") == "complete":
            raise RuntimeError("C3 is already complete; use aggregate to rebuild summaries")
    with (run_root / "run.log").open("a", buffering=1) as log:
        command = [sys.executable, str(Path(__file__).resolve()), "execute", "--run-root",
                   str(run_root), "--workers", str(workers)]
        process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(run_root / "launch.json", dict(launched_at=now(), pid=process.pid,
                                             command=command, log=str(run_root / "run.log")))
    return dict(pid=process.pid, run_root=str(run_root), log=str(run_root / "run.log"),
                workers=workers, state="launched; check status for fitting progress")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "launch", "status", "aggregate"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.run_root)
        print(json.dumps({key: value[key] for key in ("created_at", "n_cells", "n_quantile_fits")}, indent=2))
    elif args.command == "status":
        print(json.dumps(status(args.run_root), indent=2))
    elif args.command == "launch":
        print(json.dumps(launch(args.run_root, args.workers), indent=2))
    elif args.command == "aggregate":
        from opal2.quantile_direct_evaluation import aggregate
        print(aggregate(PROJECT, args.run_root))
    else:
        coordinator(args.run_root, args.workers)


if __name__ == "__main__":
    main()
